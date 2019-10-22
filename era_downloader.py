#!/usr/bin/env python
"""Some convenience files to grab meteorological data and convert
to CABO format to use within WOFOST. So far, using ERA5
"""
import struct
import logging 

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import datetime as dt

from textwrap import dedent

from collections import namedtuple

from pathlib import Path

import numpy as np

from osgeo import gdal

from netCDF4 import Dataset, date2index

import cdsapi


ERAPARAMS = namedtuple("ERAPARAMS", ["ssrd", "mx2t", "mn2t", "tp", 
                                     "u10", "v10", "d2m"])
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - ' +
                                  '%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    LOG.addHandler(ch)
LOG.propagate = False

def grab_era5(year, output_folder, region, mylat, mylon):
    """A function to downlaod the ERA5 data for one year for a given location.
    Note that this takes a while! Also, you need to have the Copernicus Data
    Service API configured for this to work.
    
    Parameters
    ----------
    output_fname : str
        The output file name. This will be a netCDF file
    year : int
        The year of interest. Must lie within the ERA5 years!
    mylat : float
        Latitude in decimal degrees.
    mylon : float
        Longitude in decimal degrees.
    """
    #'80/-50/-25/0', # North, West, South, East.
    area = f"{int(mylat[0]):d}/{int(mylon[0]):d}/" + \
           f"{int(mylat[1]):d}/{int(mylon[1]):d}"
    #[60, -10, 50, 2], # North, West, South, East
    c = cdsapi.Client()                
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable':[
                'surface_solar_radiation_downwards',
                'maximum_2m_temperature_since_previous_post_processing',
                'minimum_2m_temperature_since_previous_post_processing',
                'total_precipitation',
                '10m_u_component_of_wind','10m_v_component_of_wind',
                '2m_dewpoint_temperature'                            
            ],
            'product_type':'reanalysis',
            'year':f"{year:d}",
            'month':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12'
            ],
            'day':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12',
                '13','14','15',
                '16','17','18',
                '19','20','21',
                '22','23','24',
                '25','26','27',
                '28','29','30',
                '31'
            ],
            'time':[
                '00:00','01:00','02:00',
                '03:00','04:00','05:00',
                '06:00','07:00','08:00',
                '09:00','10:00','11:00',
                '12:00','13:00','14:00',
                '15:00','16:00','17:00',
                '18:00','19:00','20:00',
                '21:00','22:00','23:00'
            ],
            #'area' : "%d/%d/%d/%d"%
            'area' : area,
            'format':'netcdf'
        },
        (Path(output_folder)/f"{region:s}_{year:d}.nc").as_posix())

        
if __name__ == "__main__":
    mylat = [23, -35]
    mylon = [-23, 42]
    output_folder = "/data/geospatial_08/ucfajlg/ERA5_meteo"
    wrapper = partial (grab_era5, region="Africa", output_folder=output_folder, 
                       mylat=mylat, mylon=mylon)
    
    # create a thread pool of 4 threads
    years = np.arange(2013,2020).astype(np.int)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for _ in executor.map(wrapper, years):
            pass
