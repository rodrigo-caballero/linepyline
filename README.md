# linepyline

A line-by-line radiative transfer model in pure python---no C or
Fortran extensions to compile. The code runs fast thanks to [numba](https://numba.readthedocs.io) parallelization, which
gives order-of-magnitude speedup on multi-core CPUs.

It comes with [HITRAN 2024 line lists](http://hitran.org/) and the [MTCKD 4.3 continuum
model](http://rtweb.aer.com/continuum_frame.html)  preinstalled. 

It is geared to climate-oriented problems in Earth and planetary atmospheres. You can
specify the atmosphere to consist of an arbitrary mixture of absorbing gases

# Quickstart
A quick example. See `examples/US-standard-atmosphere-example.ipynb` for more details
```
import xarray as xr
from matplotlib import pyplot as plt
import linepyline as lpl

# instantiate a linepyline radiative transfer model 
rtm = lpl.rtm()

# open file containing US Standard Atmosphere data and set 
atm = xr.open_dataset('afgl_1986-us_standard.nc')

# set profiles
p = atm.p # pressure coordinate, must be in Pa and ordered by increasing p
ps = p.isel(p=-1) # surface pressure
T = atm.t # atmospheric temperature, must be in K
Ts = T.isel(p=-1) # surface (skin) temperature

# concentration of radiatively active species (must be molar fraction, units ppv)
# try with water vapor only
absorbers = {'H2O' : atm.x_H2O}

# transparent background gas mixed in with absorbers
background_gas = 'air'

#spectral resolution and range(cm-1) 
dnu = 0.1 
nu_min = dnu
nu_max = 2000

# line profile to use
line_shape = 'pseudovoigt'

# do the calculation; all output stored in xarray Dataset ds
# runtime is 0.4 s on an 8-core MacBook M3
ds = rtm.radiative_transfer(nu_min, nu_max, dnu, p, ps, T, Ts, absorbers=absorbers,
background_gas=background_gas, line_shape=line_shape)

# make a spectrally-coarsend version of the output (averages over blocks of width in cm-1)
ds_coarse = rtm.coarsen(ds, dnu, width=10)

# plot
plt.plot(ds.nu, ds.olr, 'k,', alpha=0.2, label='OLR')
plt.plot(ds_coarse.nu, ds_coarse.olr, label='OLR coarse')
plt.plot(ds.nu, ds.lw_up_srf, 'k:', label='Surface upward')
plt.legend()
plt.gca().set_xlabel('wavenumber (cm-1)')
plt.gca().set_ylabel('LW flux W/m2/cm-1');
```
![](examples/h2o_only.svg)
```
# try it again, including CO2
# if concentration given as scalar, will be assumed uniform (well mixed) through column
absorbers = {'H2O' : atm.x_H2O,
                   'CO2' : 400*1.e-6}

# this one takes 0.9 s 
ds = rtm.radiative_transfer(nu_min, nu_max, dnu, p, ps, T, Ts, a
```
![](examples/h2o_co2.svg)

# Installation

-
   [Download](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github) or
   [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository?tool=webui)
   this repository

- Install
```
cd linepyline
pip install .
```

# Dependencies

linepyline depends on  (version in parentheses used in developing/testing
the project):
```
python (3.12)
numpy (2.26)
xarray (2025.4.0)
scipy (1.15.1)
numba (0.63.1)
[numba-stats](https://github.com/scikit-hep/numba-stats) (1.11)
```
Using conda, you can install these into your current environment:
```
conda install -c conda-forge numpy xarray scipy numba numba-stats
```
or create a new environment
```
conda create -n linepyline -c conda-forge python=3.12 numpy xarray scipy numba numba-stats
```
```





