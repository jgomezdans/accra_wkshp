# Earth Observation, Crop Modelling & Data Assimilation workhop

## National Centre for Earth Observation (NCEO, UK) & GSSTI (Ghana)

<div style="float:right">
<table>
<tr>
    <td> 
        <img src="figs/nceo_logo.png" alt="NCEO logo" style="width:200px;height:40px;"/> 
    </td>
    <td> 
        <img src="figs/gssti_logo.png" alt="GSSTI logo" style="width:200px;height:40px;"/> 
    </td>
    <td> 
        <img src="figs/multiply_logo.png" alt="H2020MULTIPLY logo" style="width:40px;height:40px;"/> 
    </td>
</tr>
</table>
</div>


#### J Gomez-Dans (NCEO & UCL) `j.gomez-dans@ucl.ac.uk`

This repository contains a number of Jupyter Python notebooks that demonstrate accessing datasets, including meteo and EO data, developing and running crop models, as well as deploying data assimilation systems to monitor crop growth.


### Running the notebooks on the browser

* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jgomezdans/accra_wkshp/1.2?filepath=01-Meteo_Crop_Exploration.ipynb) A brief exploration of meteorological data from an agroclimatology perspective.
* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jgomezdans/accra_wkshp/1.2?filepath=02-MODIS_LAI_exploration.ipynb) Exploring MODIS LAI data products over Ghana.
* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jgomezdans/demo_ghana/master?filepath=examine_data.ipynb) A brief illustration of Sentinel-2 data over northern Ghana.
* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jgomezdans/accra_wkshp/1.2?filepath=03-Production_Efficiency_Modelling.ipynb) This notebook develops the intuition of a very simple production efficiency model (PEM).
* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jgomezdans/accra_wkshp/1.2?filepath=04-WOFOST_playground.ipynb) A notebook demonstrating the use of the WOFOST crop model, applied to maize in northern Ghana.
* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jgomezdans/accra_wkshp/1.2?filepath=05-DA_wofost.ipynb) Using data assimilation (DA) with crop growth models (WOFOST), an example using the Ensemble Kalman Filter (EnKF)



### Installing on your own computer

If you want to install this on your own computer, you can either close or download the repository, install the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [Anaconda](https://www.anaconda.com/distribution/)) python distribution, and you can install all the required packages using

```
conda env create -f environment.yml
```

This will install your own environment.

