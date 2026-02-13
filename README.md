# linepyline

A line-by-line radiative transfer model in pure python. The code is fast thanks to
[numba](https://numba.readthedocs.io) parallelization, which gives order-of-magnitude
speedup on multi-core CPUs.

linepyline is aimed at climate-oriented problems in Earth and planetary atmospheres. You
can specify the atmosphere to consist of an arbitrary mixture of absorbing gases and a
transparent background gas like dry Earth air or pure N2. The absorbing gases do not
need to be dilute -- if fact the background gas can be entirely absent.

linepyline defines a single class, `linepyline.rtm()`,  providing functions to compute
mass absorption coefficients, optical depth for the absorbers, and solve for clear-sky
longwave (thermal) radiative fluxes using the 2-stream approximation. There is currently
no treatment for absorption by clouds, or shortwave absorption and scattering.

Iinepyline comes with [HITRAN 2024](http://hitran.org/) line lists for the main isotopes
of H2O, CO2, O3, CH3 and NH3, and with the [MTCKD 4.3 water vapor continuum
model](http://rtweb.aer.com/continuum_frame.html), all preinstalled in netCDF format.

You can download more line lists using the Hitran API
([HAPI](https://github.com/hitranonline/hapi)), which is included in the linepyline
distribution. The script `linepyline/HITRAN/HAPI/download_HITRAN_to_netcdf.py` shows you
how to download the line data and convert it to netCDF. You may also need to add
thermodynamic data for the new molecules to the gases inventory in `linepyline/phys.py`.

## How it works
A quick example to illustrate basic usage. 
```
import xarray as xr
import linepyline

# Initialize a linepyline radiative transfer model (rtm) object
rtm = linepyline.rtm(background_gas='air', use_numba=True)

# Open file containing US Standard Atmosphere data for this example
atm = xr.open_dataset('afgl_1986-us_standard.nc')

# Set thermodynamic profiles
p = atm.p           # total atmospheric pressure (vertical coordinate), 
                         # must be in Pa and ordered by increasing p
T = atm.t           # atmospheric temperature, must be in K
ps = p.isel(p=-1) # surface pressure (Pa)
Ts = T.isel(p=-1) # surface (skin) temperature (K)

# Set concentration of radiatively active species (must be molar fraction, units ppv)
# Water vapor only in this case
absorbers = {'H2O' : atm.x_H2O}

# Set spectral resolution and range (cm-1) 
dnu = 0.1 
nu_min = dnu
nu_max = 2000

# Set line profile to use
line_shape = 'pseudovoigt'

# Compute mass absorption coefficients, optical depth and thermal radiative fluxes.
# Runtime for this call on an 8-core MacBook M3 is 
# 0.4 s with numba, 10.6 s without numba, ~25x speedup
ds = rtm.radiative_transfer(nu_min, nu_max, dnu, p, ps, T, Ts, \
       absorbers=absorbers, background_gas=background_gas, line_shape=line_shape)

# Make a spectrally-coarsend version of the output 
# (averages over blocks of width in cm-1)
ds_coarse = rtm.coarsen(ds, dnu, width=10)

# Plot
```
![](examples/h2o_only.svg)
```
# Try it again, now including CO2
absorbers = {'H2O' : atm.x_H2O,
                   'CO2' : 400*1.e-6}

# Repeat the calculation. This one takes ~1 s -- CO2 has a lot of lines
ds = rtm.radiative_transfer(nu_min, nu_max, dnu, p, ps, T, Ts, a
```
![](examples/h2o_co2.svg)

## Installation

-
   [Download](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github) or
   [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository?tool=webui)
   this repository

- Install:
```
cd linepyline
pip install .
```

## Dependencies

linepyline requires  (version in parentheses used in developing/testing
the project):
```
python (3.12)
numpy (2.26)
xarray (2025.4.0)
netCDF4 (1.7.2)
scipy (1.15.1)
numba (0.63.1)
numba-stats (1.11)
```
Using conda, you can install these into your current environment:
```
conda install -c conda-forge numpy xarray scipy numba numba-stats
```
or create a new environment
```
conda create -n linepyline -c conda-forge python=3.12 numpy xarray netCDF4 scipy numba numba-stats matplotlib
```

## Acknowledgements
linepyline was inspired by Daniel Koll's [PyRADS](https://github.com/danielkoll/PyRADS)
and by Ray Pierrehumbert's book [*Principles of Planetary Climate*](https://geosci.uchicago.edu/~rtp1/PrinciplesPlanetaryClimate) and related python code.




