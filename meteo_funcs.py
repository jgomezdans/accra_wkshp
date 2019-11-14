#!/usr/bin/env python
"""Some functionality for playing around with the meteo data"""

import datetime as dt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

import ipywidgets as widgets
from IPython.display import display

from process_meteo_drivers import grab_meteo_data

parameters = ["Irradiance \n(kJ·m-2·d-1)",
     "Min temp \n(degC)",
    "Max temp \n (degC)",
    "Vap press\n(kPa)",
    "Wind spd \n (m·s-1)",
    "Precip\n (mm·d-1)"]


regions = ["Ashanti",  "Brong_Ahafo",  "Central",  
           "Eastern",  "Greater_Accra",  "Northern",
           "Upper_East",  "Upper_West",  "Volta",
           "Western"]


def aggregate_plots():
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)

    dates = pd.date_range(start_date, end_date, freq='D')

    options = [(date.strftime(' %d %b %Y '), date) for date in dates]
    index = (0, len(options)-1)

    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Sowing & Harvest',
        orientation='horizontal',
        layout={'width': '600px'}
    )

    def plot_aggr_meteo(sowing_harvesting, region_name, selected_years):
        sowing, harvesting = sowing_harvesting
        meteo_files = get_region_data_func(region_name, selected_years, do_plot=False)
        data = aggregate_meteo(meteo_files, sowing, harvesting, aggr=np.sum)
        fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True,
                figsize=(12,12), squeeze=True)
        axs = axs.flatten()
        for i in range(6):
            axs[i].plot(data[:, 0], data[:, i + 1], '-o')
            axs[i].set_title(parameters[i])
        fig.suptitle(region_name)

    widgets.interact_manual(
        plot_aggr_meteo,
        sowing_harvesting=selection_range_slider,
        region_name=widgets.Dropdown(
                        options=regions, value='Central', description='Region:',
                        disabled=False,), 
        selected_years=widgets.IntRangeSlider(min=2010, max=2018, value=(2015,2016)))



def plot_meteo(meteo):
    """Plot WOFOST meteo files
    
    Parameters
    ----------
    meteo : str or iter
        Set of text files that contain different variables of interest
        to plot. Can be done with just a file or a list of files.
    """

    if type(meteo) != type([]): meteo = [meteo]
    
    fig, axs = plt.subplots(nrows=3, ncols=2, 
                figsize=(13,9), sharex=True,squeeze=True)
    axs = axs.flatten()
    for meteo_file in meteo:
        d = np.loadtxt(meteo_file.as_posix(), skiprows=20)
        for i,p in enumerate(parameters):
            if i == 5:
                axs[i].plot(d[:,2], d[:,3+i], '-', lw=0.8, label=meteo_file.name)
            else:
                axs[i].plot(d[:,2], d[:,3+i], '-', lw=0.8)
            axs[i].set_ylabel(p, fontsize=9)
    axs[-1].legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()


def calc_et0(r_surf, t_min, t_max):
    """Calculate Hargreaves ET0 in mm/day
    
    Parameters
    ----------
    r_surf : float, array
        Surface radiation
    t_min : float, array
        Min daily temperature (degC)
    t_max : float array
        Max daily temperature (degC)
    """
    t_mean = 0.5*(t_min + t_max)
    lam = 2260.
    et0 = 0.0023 * ((t_max - t_min)** 0.5) * (t_mean + 17.8) * r_surf / lam
    return et0



def aggregate_meteo(meteo_files, sowing, harvesting, aggr=np.cumsum):
    rr = []
    for meteo_file in meteo_files:
        year = int(meteo_file.name.split(".")[-1])
        d = np.loadtxt(meteo_file.as_posix(), skiprows=20)
        doy = np.array([
            dt.datetime(year, 1, 1) + dt.timedelta(days=int(j))
            for j in d[:, 2]])
        sow = dt.datetime(year, sowing.month, sowing.day)
        harvest = dt.datetime(year, harvesting.month, harvesting.day)

        passer = np.logical_and(doy >= sow,
                                doy <= harvest)
        xx = aggr(d[passer, 3:], axis=0)
        rr.append(np.r_[year, xx])
    rr = np.array(rr)
    return rr


def extract_data(lat, lon, meteo_folder="era5_data",
                 n_threads=2):
    meteo_files = []
    wrapper = lambda year: grab_meteo_data(
            lon,
            lat,
            year,
            f"era5_africa_{year:d}.nc",
            meteo_folder,
            site_name="Ghana")
    # create a thread pool of n_threads threads

    years = np.arange(2010,2019).astype(np.int)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for _ in executor.map(wrapper, years):
            pass


def get_region_data():
    @widgets.interact(region_name=widgets.Dropdown(
                            options=regions, value='Central', description='Region:',
                            disabled=False,), 
                            selected_years=widgets.IntRangeSlider(min=2010, max=2018, value=(2015,2016)))
    def get_region_data_fun(region_name, selected_years, do_plot=True):
        start_year, end_year = selected_years
        meteo_files = sorted([f for f in Path(f'./data/meteo/{region_name}/').glob(f"{region_name}.20??")])
        years = [int(f.name.split(".")[1]) for f in meteo_files]
        do_files = [f for y, f in zip(years, meteo_files) if start_year <= y <= end_year]
        if do_plot:
            plot_meteo(do_files)
        else:
            return do_files
        
        
def get_region_data_func(region_name, selected_years, do_plot=True):
    start_year, end_year = selected_years
    meteo_files = sorted([f for f in Path(f'./data/meteo/{region_name}/').glob(f"{region_name}.20??")])
    years = [int(f.name.split(".")[1]) for f in meteo_files]
    do_files = [f for y, f in zip(years, meteo_files) if start_year <= y <= end_year]
    if do_plot:
        plot_meteo(do_files)
    else:
        return do_files