#!/usr/bin/env python

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



def set_up_wofost(crop_start_date, crop_end_date,
                  meteo, crop, variety, soil,
                  wav=100, co2=400, rdmsol=100.):
    cropdata = YAMLCropDataProvider(fpath="./WOFOST_crop_parameters")
    cropdata.set_active_crop(crop, variety)
    soildata = CABOFileReader(soil)
    soildata['RDMSOL'] = rdmsol
    sitedata = WOFOST71SiteDataProvider(WAV=wav, CO2=co2)
    parameters = ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata)
    with open("temporal.amgt", 'w') as fp:
        fp.write(agromanagement_contents.format(year=crop_start_date.year,
                        crop=crop, variety=variety, crop_start_date=crop_start_date,
                        crop_end_date=crop_end_date))
    agromanagement = YAMLAgroManagementReader("temporal.amgt")

    wdp = CABOWeatherDataProvider(meteo, fpath=f"./data/meteo/{meteo}/")
    return parameters, agromanagement, wdp
    

def run_wofost(parameters, agromanagement, wdp, potential=False):
    if potential:
        wofsim = Wofost71_PP(parameters, wdp, agromanagement)
    else:
        wofsim = Wofost71_WLP_FD(parameters, wdp, agromanagement)
    wofsim.run_till_terminate()
    df_results = pd.DataFrame(wofsim.get_output())
    df_results = df_results.set_index("day")
    return df_results, wofsim


def change_sowing_date(start_sowing, end_sowing, meteo, crop, variety, soil, mgmt,
                        n_days=10):
    fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, squeeze=True,
                           figsize=(16,16))
    axs = axs.flatten()
    sowing_date = start_sowing
    
    while sowing_date < end_sowing:
        parameters, agromanagement, wdp = set_up_wofost(
                sowing_date, sowing_date + dt.timedelta(days=150),
                meteo, crop, variety, soil)
        
        df_results, simulator = run_wofost(parameters, agromanagement, wdp,
                                           potential=False)
        sowing_date += dt.timedelta(days=n_days)
        for j, p in enumerate(WOFOST_PARAMETERS):
            axs[j].plot_date(df_results.index, df_results[p], '-')
            axs[j].set_ylabel(WOFOST_LABELS[p], fontsize=8)
    # fig.autofmt_xdate()

    plt.gcf().autofmt_xdate()
    plt.gca().fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')
    plt.xlim(start_sowing, None)
    axs[3].set_xlabel("Time [d]")

def change_sowing_slider():
    interact(change_sowing_date,
            start_sowing=widgets.DatePicker(value=dt.date(2011, 7, 1),
            description="Earliest possible sowing date"),
            end_sowing=widgets.DatePicker(value=dt.date(2011, 8, 10),
            description="Latest possible sowing date"),
            meteo=widgets.Dropdown(
                        options=regions, value='Upper_East', description='Region:',
                        disabled=False,),
            n_days = widgets.IntSlider(min=1, max=20, value=10),
            crop=fixed("maize"),
            variety=fixed("Maize_VanHeemst_1988"),
            soil=fixed("ec4.new"),
            mgmt=fixed("ghana_maize.amgt"))
