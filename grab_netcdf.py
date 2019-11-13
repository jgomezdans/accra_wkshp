#!/usr/bin/env python
"""Copy the netCDF files into the binder docker image
"""
from concurrent.futures import ThreadPoolExecutor

import urllib.request

urls = [f"http://www2.geog.ucl.ac.uk/~ucfajlg/ERA5_Africa/era5_africa_{year:d}.nc"
        for year in range(2010, 2020)]
        
def download_nc(url):
    fname_out = url.split("/")[-1]
    urllib.request.urlretrieve(url, fname_out)


if __name__ == "__main__":
    print("Downloading NetCDF files")
    with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(download_nc, urls)
            
    print("Done downloading netCDF files")
