# Download line data from the HITRANonline database and save it to netcdf
# See [https://hitran.org]. You might want to re-run this get the most recent version of the data

import hapi # import the HITRAN application interface
import xarray as xr
from pathlib import Path

# initialize the database that will store line data in HITRAN .par format
hapi.db_begin('data')

# download data from Hitran.org
# get main isotope only
# get all lines available

molecule_IDs = {'H2O':1, 'CO2':2,'O3':3, 'CH4':6, 'O2':7, 'NH3':11}
isotope_id = 1
wavenum_min = 0.
wavenum_max = 1.e12 # high number to ensure we get all lines
for name, ID in molecule_IDs.items():
    hapi.fetch(name, ID, isotope_id, wavenum_min, wavenum_max)

# save to netcdf
for name in molecule_IDs.keys():
    d = hapi.getTableHeader(name)
    nu_l = hapi.getColumn(name, 'nu').data
    nu_l = xr.DataArray(nu_l, coords={'nu_l':nu_l}, name='nu_l').assign_attrs({'long_name':'Transition wavenumber (cm-1)'})
    container = []
    for param in d['description'].keys():
        if param != 'nu':
            container.append(
                xr.DataArray(hapi.getColumn(name, param).data, coords={'nu_l':nu_l}, name=param).assign_attrs({'long_name':d['description'][param]})
            )
    ds = xr.merge(container)
    
    # drop string variables (they make the files very large and are not needed in linepyline)
    drop_vars = []
    for var in ds.data_vars:
        if (ds.data_vars[var].dtype != float) and (ds.data_vars[var].dtype != int):
            drop_vars.append(var)
    print(f'Dropping {drop_vars}')
    ds = ds.drop_vars(drop_vars)

    # optionally encode to further reduce size
    encoding = {}
    for var in ds.data_vars:
        if ds.data_vars[var].dtype == float:
            encoding[var] = {
                'dtype': 'int32',      
                'zlib': True,
                'complevel': 5,
                }
   
    netcdf_dir = Path('../HITRAN2024/netcdf')
    netcdf_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(netcdf_dir/f'{name}_hitran_line_data.nc') #, encoding=encoding)




