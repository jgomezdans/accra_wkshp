#!/usr/bin/env python

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from pcse.fileinput import CABOFileReader, YAMLCropDataProvider, CABOWeatherDataProvider
from pcse.fileinput import YAMLCropDataProvider, YAMLAgroManagementReader
from pcse.util import WOFOST71SiteDataProvider
from pcse.base import ParameterProvider
from pcse.models import Wofost71_WLP_FD, Wofost71_PP


def set_up_wofost(meteo, crop, variety, soil, mgmt, wav=100, co2=400, rdmsol=100.):
    cropdata = YAMLCropDataProvider(fpath="./WOFOST_crop_parameters")
    cropdata.set_active_crop(crop, variety)
    soildata = CABOFileReader(soil)
    soildata['RDMSOL'] = rdmsol
    sitedata = WOFOST71SiteDataProvider(WAV=wav, CO2=co2)
    parameters = ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata)
    agromanagement = YAMLAgroManagementReader("ghana_maize.amgt")
    wdp = CABOWeatherDataProvider(meteo, fpath="./")
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