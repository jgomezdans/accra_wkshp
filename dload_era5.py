#!/usr/bin/env python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve

def download_era_files():
    p = Path.cwd()
    pp = (p/"era5_data").mkdir(exist_ok=True, parents=True)
    urls = [f"http://www2.geog.ucl.ac.uk/~ucfajlg/ERA5_Africa/era5_africa_{year:d}.nc"
            for year in range(2010, 2020)]
    def grab_url(url):
        urlretrieve(url, f"era5_data/{url.split("/")[-1]:s}")
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in executor.map(grab_url, urls):
            pass
