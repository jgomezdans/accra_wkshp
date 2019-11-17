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

def set_up_wofost(meteo, crop, variety, soil, mgmt, wav=100, co2=400, rdmsol=100.):
    cropdata = YAMLCropDataProvider(fpath="./WOFOST_crop_parameters")
    cropdata.set_active_crop(crop, variety)
    soildata = CABOFileReader(soil)
    soildata['RDMSOL'] = rdmsol
    sitedata = WOFOST71SiteDataProvider(WAV=wav, CO2=co2)
    parameters = ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata)
    agromanagement = YAMLAgroManagementReader("ghana_maize.amgt")

    wdp = CABOWeatherDataProvider(meteo, fpath="./data/meteo/Upper_West/")
    return parameters, agromanagement, wdp
    

def run_wofost(parameters, agromanagement, wdp, potential=True):
    if potential:
        wofsim = Wofost71_PP(parameters, wdp, agromanagement)
    else:
        wofsim = Wofost71_WLP_FD(parameters, wdp, agromanagement)
    wofsim.run_till_terminate()
    df_results = pd.DataFrame(wofsim.get_output())
    df_results = df_results.set_index("day")
    return df_results, wofsim


def change_sowing_date(sowing_date, harvest_date, meteo, crop, variety, soil, mgmt,
                        n_days=10):
    parameters, agromanagement, wdp = set_up_wofost(meteo, crop, variety, soil, mgmt,
                                                    wav=0, co2=400, rdmsol=100.,)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, squeeze=True,
                           figsize=(14,14))
    axs = axs.flatten()
    while sowing_date < harvest_date:
        agromanagement[0][dt.date(
            2011, 1, 1)]['CropCalendar']['crop_start_date'] = sowing_date
        df_results, simulator = run_wofost(parameters, agromanagement, wdp,
                                           potential=False)
        sowing_date += dt.timedelta(days=n_days)
        axs[0].plot_date(df_results.index, df_results.TAGP)
        axs[1].plot_date(df_results.index, df_results.SM)
        axs[2].plot_date(df_results.index, df_results.LAI)
        axs[3].plot_date(df_results.index, df_results.DVS)
    # fig.autofmt_xdate()
    plt.gcf().autofmt_xdate()
    plt.gca().fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')
    plt.xlim(dt.date(2011, 3, 1))
    

def change_sowing_slider():
    interact(change_sowing_date,
            sowing_date=widgets.DatePicker(value=dt.date(2011, 7, 1)),
            harvest_date=widgets.DatePicker(value=dt.date(2011, 10, 1)),
            #meteo=widgets.Dropdown(
            #            options=regions, value='Upper_East', description='Region:',
            #            disabled=False,),
            meteo=fixed("Upper_West"),
            crop=fixed("maize"),
            variety=fixed("Maize_VanHeemst_1988"),
            soil=fixed("ec4.new"),
            mgmt=fixed("ghana_maize.amgt"))
