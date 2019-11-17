#!/usr/bin/env python
import copy
from pathlib import Path
import datetime as dt
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


import pandas as pd

from pcse.fileinput import CABOFileReader, YAMLCropDataProvider, CABOWeatherDataProvider
from pcse.fileinput import YAMLCropDataProvider, YAMLAgroManagementReader
from pcse.util import WOFOST71SiteDataProvider
from pcse.base import ParameterProvider
from pcse.models import Wofost71_WLP_FD, Wofost71_PP

import ipywidgets.widgets as widgets
from ipywidgets import interact, interactive, fixed


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

regions = ["Ashanti",  "Brong_Ahafo",  "Central",  
           "Eastern",  "Greater_Accra",  "Northern",
           "Upper_East",  "Upper_West",  "Volta",
           "Western"]
agromanagement_contents = """
Version: 1.0
AgroManagement:
- {year:d}-01-01:
    CropCalendar:
        crop_name: '{crop}'
        variety_name: '{variety}'
        crop_start_date: {crop_start_date}
        crop_start_type: sowing
        crop_end_date: {crop_end_date}
        crop_end_type: harvest
        max_duration: 150
    TimedEvents: null
    StateEvents: null
"""


WOFOST_PARAMETERS = ['DVS', 'LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST',
                'TWRT', 'TRA', 'RD', 'SM']
LABELS = ["Development stage [-]", "LAI [m2/m2]",
                 "Total Biomass [kg/ha]",
                 "Total Storage Organ Weight [kg/ha]",
                 "Total Leaves Weight [kg/ha]",
                 "Total Stems Weight [kg/ha]",
                 "Total Root Weight [kg/ha]",
                 "Transpiration rate [cm/d]",
                 "Rooting depth [cm]",
                 "Soil moisture [cm3/cm3]"]
WOFOST_LABELS = dict(zip(WOFOST_PARAMETERS, LABELS))



def set_wofost_up(crop="maize", variety="Maize_VanHeemst_1988",
                meteo="Upper_East", 
                soil="ec4.new", wav=60, co2=400, rdmsol=100):
    cropdata = YAMLCropDataProvider(fpath="./WOFOST_crop_parameters")
    cropdata.set_active_crop(crop, variety)
    soildata = CABOFileReader(soil)
    soildata["RDMSOL"] = rdmsol
    sitedata = WOFOST71SiteDataProvider(WAV=wav, CO2=co2)
    parameters = ParameterProvider(cropdata=cropdata,
                                   soildata=soildata, sitedata=sitedata)

    agromanagement = YAMLAgroManagementReader("ghana_maize.amgt")

    wdp = CABOWeatherDataProvider(meteo, fpath=f"./data/meteo/{meteo}/")
    return parameters, agromanagement, wdp


def prepare_observations(start_date=dt.datetime(2011, 7, 15),
                         end_date=dt.datetime(2011, 11, 1),
                        obs_file="data/sample_wofost_output.csv",
                        n_obs=7, sigma_lai=0.1, sigma_sm=0.25):

    df = pd.read_csv(obs_file)
    df['date'] = pd.to_datetime(df.day)
    df.set_index('date')
    df = df[(df.date >= start_date) &
            (df.date <= end_date)]
    lai_obs = df.LAI.values
    sm_obs = df.SM.values
    obs_times = df.date.values
    obs_passer = np.arange(len(obs_times))
    np.random.shuffle(obs_passer)
    observations = []

    # Pack them into a convenient format
    for i, (datex, lai, sm) in enumerate(zip(obs_times[obs_passer], lai_obs[obs_passer],
                                       sm_obs[obs_passer])):
        if ~np.isnan(lai) and ~np.isnan(sm):
            lai_unc = lai*sigma_lai
            sm_unc = sm * sigma_sm
            dd = datex.astype('datetime64[s]').tolist()
            observations.append((dd,
            {"LAI": (lai, lai_unc), "SM": (sm, sm_unc)}))
    observations = observations[:n_obs]

    observations = sorted(observations, key=lambda x: x[0])
    return observations


class WOFOSTEnKF(object):
    def __init__(self, assimilation_variables, override_parameters,
                n_ensemble, observations,
                lai_unc=0.1, sm_unc=0.25):
        self.n_ensemble = n_ensemble
        self.assimilation_variables = assimilation_variables
        self.override_parameters = override_parameters
        self.lai_unc = lai_unc
        self.sm_unc = sm_unc
        self.observations = observations

    def setup_wofost(self, crop="maize", variety="Maize_VanHeemst_1988",
                     meteo="Upper_East", soil="ec4.new", wav=60,
                     co2=400, rdmsol=100):
        self.parameters, self.agromanagement, self.weather_db = set_wofost_up(
            crop=crop, variety=variety,
            meteo=meteo, 
            soil=soil, wav=wav, co2=co2, rdmsol=rdmsol)
        self._setup_ensemble()

    def _setup_ensemble(self):
        self.ensemble = []
        for i in range(self.n_ensemble):
            p = copy.deepcopy(self.parameters)
            for par, distr in self.override_parameters.items():
                p.set_override(par, distr[i])
            member = Wofost71_WLP_FD(p, self.weather_db, self.agromanagement)
            self.ensemble.append(member)

    def run_filter(self):
        for obs_date, obs in self.observations:
            print(f"Assimilating {obs_date}")
            import pdb;pdb.set_trace()
            ensemble_state = self._run_wofost_gather_sates(obs_date)
            P = np.array(ensemble_state.cov().values)
            ensemble_obs = self._observations_ensemble(obs)
            R = np.array(ensemble_obs.cov().values)
            K = self.kalman_gain(obs, P, R)
            x = np.array(ensemble_state.values).T
            y = np.array(ensemble_obs.values).T
            x_opt = x + K @ (y - x)
            df_analysis = pd.DataFrame(x_opt.T,
                columns=self.assimilation_variables)
            
            for member, new_states in zip(self.ensemble,
                                        df_analysis.itertuples()):
                member.set_variable("LAI", new_states.LAI)
                member.set_variable("SM", new_states.SM)

        [member.run_till_terminate() for member in self.ensemble]
        results = [pd.DataFrame(member.get_output()).set_index("day")
                    for member in self.ensemble]
        return results


    def kalman_gain(self, obs, P, R):
        H = np.identity(len(obs))
        K = H.T @ P @ np.linalg.inv(H.T @ P @ H + R)
        return K

    def _run_wofost_gather_sates(self, date):
        [member.run_till(date) for member in self.ensemble]
        ensemble_states = []
        for member in self.ensemble:
            t = {}
            for state in self.assimilation_variables:
                t[state] = member.get_variable(state)
            ensemble_states.append(t)
        return pd.DataFrame(ensemble_states)

    def _observations_ensemble(self, observations):
        fake_obs = []
        for state_var in self.assimilation_variables:
            (value, std) = observations[state_var]
            d = np.random.normal(value, std, (self.n_ensemble))
            fake_obs.append(d)
        df_obs = pd.DataFrame(fake_obs).T
        df_obs.columns = self.assimilation_variables
        return df_obs


if __name__ == "__main__":
    observations = prepare_observations()
    np.random.seed(42)
    n_ensemble = 50
    # A container for the parameters that we will override
    override_parameters = {}
    #Initial conditions
    override_parameters["TDWI"] = np.random.normal(150., 50., (n_ensemble))
    override_parameters["WAV"] = np.random.normal(10, 5, (n_ensemble))
    # parameters
    override_parameters["SPAN"] = np.random.normal(42, 5 ,(n_ensemble))
    override_parameters["TSUM1"] = np.random.normal(900, 50 ,(n_ensemble))
    override_parameters["TSUM2"] = np.random.normal(950, 50 ,(n_ensemble))
    override_parameters["CVL"] = np.random.normal(0.72, 0.2 ,(n_ensemble))
    override_parameters["CVO"] = np.random.normal(0.71, 0.1 ,(n_ensemble))
    override_parameters["CVR"] = np.random.normal(0.68, 0.1, (n_ensemble))
    assim_vars = ["LAI", "SM"]
    
    enkf = WOFOSTEnKF(assim_vars, override_parameters, n_ensemble, observations)
    enkf.setup_wofost()
    results = enkf.run_filter()