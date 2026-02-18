import numpy as np
import xarray as xr
from pathlib import Path
from . import phys, hapi

class rtm():

    def __init__(self,
                 hitran = 'HITRAN2024',
                 mt_ckd = 'MT_CKD_H2O-4.3',
                 use_numba = True,
                 surface_gravity = 9.81,
                 background_gas = 'air'):
        '''
        Input:
        surface_gravity: Gravitational acceleration for this atmosphere, float. 
        background_gas: Name of transparent gas mixed with absorbers, string or None
          Valid values are "air" (dry Earth air), "N2" or None
          If None, there is no background gas.
        '''
        
        # list of known absorbers
        self.known_absorbers = ['H2O','CO2','O3','CH4','NH3']

        # load HITRAN line data
        data_dir = Path(__file__).parent/f'HITRAN/{hitran}/netcdf'
        print(f'Loading {", ".join(self.known_absorbers)} line data from {data_dir}')
        self.line_data = {}
        for name in self.known_absorbers:
            file = data_dir/f'{name}_hitran_line_data.nc'
            self.line_data[name] = xr.open_dataset(file)
        
        # load MT_CKD water vapor continuum
        # using data file from https://github.com/AER-RC/MT_CKD
        data_dir = Path(__file__).parent/f'MT_CKD/{mt_ckd}/data'
        print(f'Loading H2O continuum data from {data_dir}')
        file = data_dir/f'absco-ref_wv-mt-ckd.nc'
        self.cont_data = xr.open_dataset(file).rename({'wavenumbers':'nu'})

        # decide if we're using numba
        if use_numba:
            from .heavy_lifting_numba import heavy_lifting, heavy_lifting_with_binning, two_stream
        else:
            from .heavy_lifting import heavy_lifting, heavy_lifting_with_binning, two_stream
        self.heavy_lifting = heavy_lifting
        self.heavy_lifting_with_binning = heavy_lifting_with_binning
        self.two_stream = two_stream

        # make phys routines available from object
        self.phys = phys

        # set values
        self.g = surface_gravity
        self.background_gas = background_gas
 
    def get_kappa_hitran(self, mol_name, nu_min, nu_max, dnu,
                         p, T, p_self=None, 
                         line_shape='lorentz', cutoff=25., remove_plinth=True, force_lines_to_grid=False,
                         binning=False, Nbins_gamma=500, Nbins_alpha=10):
        '''
        Compute mass absorption coefficient (m2/kg) due to line spectrum 
        
        Input:
        mol_name: name of molecule (H2O, CO2 etc)
        nu_min, nu_max, dnu: wavenumber grid (cm-1)
        p: total pressure (Pa), array or scalar
        T: temperature (K), array or scalar
        p_self: partial pressure of molecule (Pa), array or scalar, used for self-broadening.
                If None, lines will only be air broadened.
        line_shape: can be lorentz, voigt or pseudovoigt
        cutoff: cut off line tails at this value (cm-1). Standard value is 25 cm-1
        remove_plinth: if True, subtract the "plinth" or "pedestal" of the line shape.
                       The plinth is defined as the Lorentz value at 25 cm-1 from the line center.
                       Must be set to True for H2O if using MT_CKD continuum, which adds
                       the plinth value. See HITRAN documentation at http://hitran.org/docs/definitions-and-units/
        force_lines_to_grid: if True, shift line center to fall exactly on nearest grid point
                             this doesnt make much difference at high pressure when lines are broad
                             but makes a big difference where line widths ~ grid resolution
        binning: if True, bins line widhts into Nbins_gamma and Nbins_alpha bins and precomputes line profiles
                 using bin-average values of gamma, alpha. This saves time (~2-3x speedup) but reduces accuracy.
                 Note that binning implicitly forces line centers to fall on a grid point (like force_lines to grid = True)
                 
        Returns:
        kappa: [pressure, wavenumber] array of mass absorption coefficient (m2/kg)
        '''
        
        assert line_shape in ['lorentz', 'voigt', 'pseudovoigt'], \
               'line_shape {line_shape}, must be one of [lorentz, voigt, pseudovoigt]'

        # convert input data to xarray where necessary
        p, T, p_self = self.format_input(p, T, p_self)

        # set default value of p_self
        if p_self is None:
            p_self = p*0.

        # select line data for relevant range [nu_min-cutoff, nu_max+cutoff]
        ds = self.line_data[mol_name].sel(nu_l = slice(nu_min-cutoff, nu_max+cutoff))

        # reference temperature for scaling
        T_ref = 296.
        
        # scale pressure (lorentzian) broadening half-widths
        # needs pressure in atm, see http://hitran.org/docs/definitions-and-units/
        p_air  = (p - p_self)/1.013e5 # Pa -> atm
        p_self = p_self/1.013e5
        gamma =  ds.gamma_air*p_air*(T_ref/T)**ds.n_air + ds.gamma_self*p_self

        # compute Doppler (gaussian) broadening half-widths
        M = phys.gases[mol_name].MolecularWeight*1.e-3 # molar mass in kg
        alpha = ds.nu_l/phys.c*np.sqrt( 2*np.log(2.)*phys.N_avogadro*phys.k*T/M )

        # temperature scaling of line strength, see http://hitran.org/docs/definitions-and-units/
        Q = self.get_partition_sum(mol_name, T)
        Qref = self.get_partition_sum(mol_name, T_ref)
        fact1 = Qref/Q
        fact2 = np.exp(-phys.h*phys.c/phys.k*ds.elower*100.*(1/T - 1/T_ref))
        fact3 = (1.- np.exp(-phys.h*phys.c/phys.k*ds.nu_l*100./T)) / \
                 (1.- np.exp(-phys.h*phys.c/phys.k*ds.nu_l*100./T_ref))
        S = ds.sw * fact1 * fact2 * fact3

        # convert line strength from (cm2/molecule)(cm-1) to (m2/kg)(cm-1) units
        M = phys.gases[mol_name].MolecularWeight*1.e-3 # molar mass in kg
        S = S * 1.e-4*phys.N_avogadro/M

        # get the target wavenumber grid
        nu = self.get_nu_grid(nu_min, nu_max, dnu)

        # now do the heavy lifting: sum contributions of absorption from each line over wavenumber grid
        # need to pass np arrays since routine is numba-ified.
        # S, gamma, alpha must be 2D with dims (p, nu_l)
        S = S.transpose(..., 'nu_l').data
        gamma = gamma.transpose(..., 'nu_l').data
        alpha = alpha.transpose(..., 'nu_l').data
        if len(S.shape) == 1:
            S = S.reshape((1,len(ds.nu_l)))
            gamma = gamma.reshape((1,len(ds.nu_l)))
            alpha = alpha.reshape((1,len(ds.nu_l)))

        if binning:
            kappa = self.heavy_lifting_with_binning(nu.data, ds.nu_l.data, S, gamma, alpha, int(Nbins_gamma), int(Nbins_alpha),
                                               cutoff, line_shape, remove_plinth)
        else:
            kappa = self.heavy_lifting(nu.data, ds.nu_l.data, S, gamma, alpha,
                                  cutoff, line_shape, remove_plinth, force_lines_to_grid)
    
        # wrap result into xarray
        kappa = xr.DataArray(kappa, coords={'p':p, 'nu':nu}, name=f'kappa_{mol_name}',
                             attrs={'long_name':f'{mol_name} mass absorption coefficient (m2/kg)'}).squeeze()

        return kappa        

    def get_kappa_mtckd(self, nu_min, nu_max, dnu, p, T, p_H2O, closure=False):
        '''
        Compute mass absorption coefficient (m2/kg) due to MT_CKD v4 continuum

        Reference (equations referenced in code are from this paper):
        Mlawer et al (2023): The inclusion of the MT_CKD water vapor continuum model in the
        HITRAN molecular spectroscopic database. Journal of Quantitative Spectroscopy & Radiative Transfer 306, 108645
        Code here reproduces what is done in MT_CKD_H2O/src/mt_ckd_h2o_module.f90
        
        Input:
        nu_min, nu_max, dnu: wavenumber grid (cm-1)
        p: total pressure (Pa), array or scalar
        T: temperature (K), array or scalar
        p_H2O: partial pressure of water vapor (Pa), array or scalar
        closure: selects between two alternative MT_CKD foreign absorption coefficients
                 if True, selects for_closure_absco, which gives radiative closure for the particular dataset used to derive the 
                          absorbption coefficients, but may be contaminated by aerosol absorption at the data collection sites
                 if False, selects for_absco, which tries to eliminate the aerosol contribution to give a true clear-sky value
                           see Mlawer et al 2024 JGR, "A more transparent infrared window"
        Returns:
        kappa: [pressure, wavenumber] array of mass absorption coefficient (m2/kg)
        '''

        # convert input data to xarray where necessary
        p, T, p_H2O = self.format_input(p, T, p_H2O)
        
        # select  data for relevant range
        # input data has resolution of 10 cm-1, so take in extra 10 cm-1 on each side
        # to make sure we can interpolate properly later
        ds = self.cont_data.sel(nu = slice(nu_min-10, nu_max+10))

        # reference pressure and temperature for scaling
        p_ref = ds.ref_press*100. # mb -> Pa
        T_ref = ds.ref_temp
        
        # density scaling (Eq. 1)
        # note that rho in that equation is number density, not mass density. here I use n to make that clear
        # note that both self and foreign scale the same way, because partial pressures scale like total pressure
        # if composition is held fixed
        n = p/phys.k/T
        n_ref = p_ref/phys.k/T_ref
        kappa_self = ds.self_absco_ref * (n/n_ref)
        if closure:
            kappa_foreign = ds.for_closure_absco_ref * (n/n_ref)
        else:
            kappa_foreign = ds.for_absco_ref * (n/n_ref)

        # temperature scaling, Eq. 6 (only applies to kappa_self)
        kappa_self = kappa_self * (T_ref/T)**ds.self_texp

        # add the self and foreign coefficients weighted by number fractions to get total kappa
        kappa = kappa_self*p_H2O/p + kappa_foreign*(p - p_H2O)/p
        
        # convert units from  (cm2/molecule)(cm-1) to (m2/kg)(cm-1)
        M = phys.gases['H2O'].MolecularWeight*1.e-3 # molar mass in kg
        kappa = kappa * 1.e-4*phys.N_avogadro/M

        # radiation factor R (Eqs. 2,3)
        R = ds.nu*np.tanh(phys.h * phys.c * ds.nu*100./ 2/ phys.k / T)
        kappa = kappa * R # kappa how has units m2/kg

        # get the target wavenumber grid and interpolate
        nu = self.get_nu_grid(nu_min, nu_max, dnu)
        kappa = kappa.interp(nu=nu, method='linear')
        
        return kappa.assign_attrs({'long_name':f'H2O continuum mass absorption coefficient (m2/kg)'}).squeeze()

    def get_optical_depth(self, nu_min, nu_max, dnu,
                          p, ps, T, q=None, RH=None, absorbers=None, 
                          line_shape='lorentz', cutoff=25., force_lines_to_grid=False,
                          binning=False, Nbins_gamma=500, Nbins_alpha=10,
                          include_mtckd_continuum=True, closure=False):

        '''
        Compute optical depth given absorber concentrations

        Input:
        p: total pressure (Pa), array
        ps: total surface pressure (Pa), scalar
        T: temperature (K), array
        Ts: surface temperature (K), scalar
        absorbers: absorber molar fractions, dict of scalars or arrays
          The dictionary must have the form
          {"H2O":<value>, "CO2":<value>, etc} with <value> giving concentration in units of ppv.
          Valid species: H2O, CO2, O3, CH4, NH3.
          If concentration specified as scalar, it is assumed well-mixed (uniform through column)
        q: specific humidity (kg H2O/kg moist gas), array
          If given, overrides H2O entry in absorbers
        RH: relative humidity (fraction), scalar or array.
          If scalar, assume uniform through column.
          If given, overrides q and H2O entry in absorbers
        line_shape: which line shape to use, string. Can be "lorentz", "voigt" or "pseudovoigt".
          "pseudovoigt" is an approximation to the full Voigt profile using a linear combination of
          Lorentzian and Gaussian profiles, accurate to better than 1.2%.
          This saves a lot of time (~2x speedup compared with full Voigt profile)
        cutoff: scalar. Cut off line tails at this value (cm-1). Standard value is 25 cm-1
        force_lines_to_grid: if True, shift line center to fall exactly on nearest grid point
          This doesnt make much difference at high pressure when lines are broad
          but makes a big difference where line widths ~ grid resolution
        binning: If True, bins Lorentz and Doppler line widths (into Nbins_gamma and Nbins_alpha bins respectively),
          and precomputes line profiles using bin-average line widths. This saves time (~2-3x speedup) but reduces accuracy,
          especially at low pressure/narrow linewidths in the upper atmosphere.
          Use with caution!
        include_mtckd_continuum: Whether to include H2O continuum absorption in calculation
        closure: Selects between two alternative MT_CKD foreign absorption coefficients:
          if True, selects for_closure_absco, which gives radiative closure for the particular dataset used to derive the 
          absorbption coefficients, but may be contaminated by aerosol absorption at the data collection sites
          if False, selects for_absco, which tries to eliminate the aerosol contribution to give a true clear-sky value
          See Mlawer et al 2024 JGR, "A more transparent infrared window"

        Returns:
        optical depth (defined 0 at top of atmosphere)
        '''

        # check input and convert data to xarray where necessary
        p, ps, T, q, RH = self.format_input(p, ps, T, q, RH)
        if absorbers is not None:
            for name in absorbers:
                assert name in self.known_absorbers, f'Absorber {name} not in {self.known_absorbers}'
                absorbers[name] = self.format_input(p, absorbers[name])[1]

        # remove absorbers with zero or None concentration
        if absorbers is not None:
            entries_to_remove = []
            for name in absorbers:
                f = absorbers[name]
                if (f*p == 0).all() or (f is None):
                    entries_to_remove.append(name)
            for name in entries_to_remove:
                absorbers.pop(name)
            if len(absorbers) == 0:
                absorbers = None
                
        # create new absorbers dict which will contain molar fraction, partial pressure and specific density of each species
        # (need partial pressure for pressure scaling of line widths, and specific density to compute mass absorption coefficient)
        self.absorbers = {}
            
        # first, compute partial pressure
        if absorbers is not None:
            for name in absorbers:
                f = absorbers[name] # molar fraction
                if isinstance(f, float):
                    f = xr.full_like(p, f) # if scalar, assume uniform through column
                pp = p*f # partial pressure
                self.absorbers[name] = {'f':f, 'pp':pp}
                if name == 'H2O':
                    rh = pp/phys.satvp(T, formula='simple')
                    self.absorbers['H2O']['RH'] = rh

        # catch cases where humidity specified using q or RH
        if q is not None:
            if 'H2O' in self.absorbers: # remove H2O if already present, new value will be set
                self.absorbers.pop('H2O')
            f = self.get_number_fraction_from_specific_humidity(q, self.background_gas, self.absorbers)
            pp = p*f
            rh = pp/phys.satvp(T, formula='simple')
            self.absorbers['H2O'] = {'f':f, 'pp':pp, 'RH':rh}
            
        if RH is not None:
            if 'H2O' in self.absorbers: # remove H2O if already present, new value will be set
                self.absorbers.pop('H2O')
            if isinstance(RH, float): 
                RH = xr.full_like(p, RH) # if scalar, assume uniform through column 
            pp = RH*phys.satvp(T, formula='simple')
            f = pp/p
            self.absorbers['H2O'] = {'f':f, 'pp':pp, 'RH':RH}

        # sanity check: at this point, we should have at least one absorber in self.absorbers,
        #               and they should take up a total number fraction <= 1
        #               (if < 1, remaining number fraction is assumed to be background gas)
        assert len(self.absorbers) > 0, 'You need to give some absorbers in input!'
        f_tot = 0 # total no. fraction of absorbers
        for name in self.absorbers:
            f_tot += self.absorbers[name]['f']
        assert (f_tot <= 1).all(), 'Total number fraction exceeds 1!'
            
        # now compute specific densities
        mean_mol_weight = 0
        for name in self.absorbers:
            mean_mol_weight += self.absorbers[name]['f'] * phys.gases[name].MolecularWeight
        if self.background_gas is not None:
            mean_mol_weight += (1 - f_tot) * phys.gases[self.background_gas].MolecularWeight 
        for name in self.absorbers:
            eps = phys.gases[name].MolecularWeight / mean_mol_weight 
            self.absorbers[name]['q'] = eps * self.absorbers[name]['f']

        # also compute for background gas
        if self.background_gas is not None:
            f = 1 - f_tot
            pp = p*f
            eps = phys.gases[self.background_gas].MolecularWeight / mean_mol_weight
            q = eps*f
            self.background_gas_concentration = {'pp':pp, 'f':f, 'q':q}

        # compute mass absorption coefficient for each absorbing species
        kappas = {}
        for name in self.absorbers:
            if name == 'H2O':
                remove_plinth = include_mtckd_continuum # if include_mtckd_continuum is True, then plinth gets removed
                kappas['H2O'] = self.get_kappa_hitran('H2O', nu_min, nu_max, dnu, p, T, p_self=self.absorbers['H2O']['pp'], 
                                                      line_shape=line_shape, cutoff=cutoff,
                                                      remove_plinth=remove_plinth, force_lines_to_grid=force_lines_to_grid,
                                                      binning=binning, Nbins_gamma=Nbins_gamma, Nbins_alpha=Nbins_alpha)
                # add continuum
                if include_mtckd_continuum:
                    kappas['H2O'] += self.get_kappa_mtckd(nu_min, nu_max, dnu, p, T, self.absorbers['H2O']['pp'], closure=closure)
            else:
                kappas[name] = self.get_kappa_hitran(name, nu_min, nu_max, dnu, p, T, p_self=self.absorbers[name]['pp'], 
                                                     line_shape=line_shape, cutoff=cutoff,
                                                     remove_plinth=False, force_lines_to_grid=force_lines_to_grid,
                                                     binning=binning, Nbins_gamma=Nbins_gamma, Nbins_alpha=Nbins_alpha)

        # add up kappas weighted by specific densities to get total absorbption coefficient
        kappa = 0
        for name in self.absorbers:
            kappa += kappas[name] * self.absorbers[name]['q']

        # save quantities for output
        self.kappa = kappa
        self.kappas = kappas

        # now integrate to get optical thickness (on layer interfaces)
        # use pure numpy here
        p = p.data
        p_int = np.concatenate( ([0], (p[1:] + p[:-1])/2, [ps]) ) # interface pressure
        dp = np.diff(p_int)
        tau_int = np.zeros((len(p_int), kappa.shape[1]))
        tau_int[1:] = np.cumsum(kappa.data*dp[:,None]/self.g, axis=0)
        self.tau_int = tau_int
        
        # output tau on midpoints
        tau = (tau_int[1:] + tau_int[:-1])/2
        tau = xr.DataArray(tau, coords=kappa.coords, name='tau', attrs={'long name':'optical depth'})
        return tau

    def radiative_transfer(self, nu_min, nu_max, dnu,
                           p, ps, T, Ts, q=None, RH=None, absorbers=None, 
                           D = 1.5, line_shape='lorentz', cutoff=25., force_lines_to_grid=False,
                           binning=False, Nbins_gamma=500, Nbins_alpha=10,
                           include_mtckd_continuum=True, closure=False):                   
        '''
        Compute longwave (thermal) radiative fluxes given absorber concentrations

        Input:
        p: total pressure (Pa), array
        ps: total surface pressure (Pa), scalar
        T: temperature (K), array
        Ts: surface temperature (K), scalar
        absorbers: absorber molar fractions, dict of scalars or arrays
          The dictionary must have the form
          {"H2O":<value>, "CO2":<value>, etc} with <value> giving concentration in units of ppv.
          Valid species: H2O, CO2, O3, CH4, NH3.
          If concentration specified as scalar, it is assumed well-mixed (uniform through column)
        q: specific humidity (kg H2O/kg moist gas), array
          If given, overrides H2O entry in absorbers
        RH: relative humidity (fraction), scalar or array.
          If scalar, assume uniform through column.
          If given, overrides q and H2O entry in absorbers
        D: Elsasser diffusivity for 2-stream calculation, inverse of average zenith angle, scalar
        line_shape: which line shape to use, string. Can be "lorentz", "voigt" or "pseudovoigt"
          "pseudovoigt" is an approximation to the full Voigt profile using a linear combination of
          Lorentzian and Gaussian profiles, accurate to better than 1.2%.
          This saves a lot of time (~2x speedup compared with full Voigt profile)
        cutoff: scalar. Cut off line tails at this value (cm-1). Standard value is 25 cm-1
        force_lines_to_grid: if True, shift line center to fall exactly on nearest grid point
          This doesnt make much difference at high pressure when lines are broad
          but makes a big difference where line widths ~ grid resolution
        binning: If True, bins Lorentz and Doppler line widths (into Nbins_gamma and Nbins_alpha bins respectively),
          and precomputes line profiles using bin-average line widths. This saves time (~2-3x speedup) but reduces accuracy,
          especially at low pressure/narrow linewidths in the upper atmosphere.
          Use with caution!
        include_mtckd_continuum: Whether to include H2O continuum absorption in calculation
        closure: Selects between two alternative MT_CKD foreign absorption coefficients:
          if True, selects for_closure_absco, which gives radiative closure for the particular dataset used to derive the 
          absorbption coefficients, but may be contaminated by aerosol absorption at the data collection sites
          if False, selects for_absco, which tries to eliminate the aerosol contribution to give a true clear-sky value
          See Mlawer et al 2024 JGR, "A more transparent infrared window"
        '''

        # convert input data to xarray where necessary
        p, ps, T, Ts, q, RH = self.format_input(p, ps, T, Ts, q, RH)

        # get optical depth
        tau = self.get_optical_depth(nu_min, nu_max, dnu, 
                                     p, ps, T, q=q, RH=RH, absorbers=absorbers, 
                                     line_shape=line_shape, cutoff=cutoff, closure=closure, force_lines_to_grid=force_lines_to_grid,
                                     binning=binning, Nbins_gamma=Nbins_gamma, Nbins_alpha=Nbins_alpha,
                                     include_mtckd_continuum=include_mtckd_continuum)
      
        # get source functions on target grid
        nu = self.get_nu_grid(nu_min, nu_max, dnu)
        B = self.Planck(nu, T)
        Bs = self.Planck(nu, Ts)
    
        # compute 2-stream fluxes on layer interfaces, defined positive upward
        # note self.tau_int is the optical depth on interfaces, updated in the previous call to get_optical_depth
        Fup_srf, Fup_atm, Fup, Fdn = self.two_stream(B.data, Bs.data, self.tau_int, D)
        Fnet = Fup + Fdn

        # heating rate (K/day/cm-1)
        # first need to compute heat capacity of gas mixture as mass-weighted mean Cp of mixture
        cp = 0
        for name in self.absorbers:
            cp += self.absorbers[name]['q'] * phys.gases[name].cp
        if self.background_gas is not None:
            cp +=  self.background_gas_concentration['q'] * phys.gases[self.background_gas].cp
        # now we can compute heating rate
        p_int = np.concatenate( ([0], (p.data[1:] + p.data[:-1])/2, [ps]) ) # interface pressure
        hr = self.g/cp.data[:,None] * np.diff(Fnet, axis=0)/np.diff(p_int)[:,None] * 86400

        # prepare output dataset
        output = [
            T.rename('T').assign_attrs({'long_name':'temperature (K)'}),                   
            xr.DataArray( Ts, name='Ts', attrs={'long_name': 'surface temperature (K)'}),
            xr.DataArray( ps/100, name='ps', attrs={'long_name': 'surface pressure (hPa)'})
            ]
        if 'H2O' in self.absorbers:
            output.append(self.absorbers['H2O']['q'].rename('q').assign_attrs({'long_name': 'specific humidity (kg/kg)'}))
            output.append(self.absorbers['H2O']['RH'].rename('RH').assign_attrs({'long_name': 'relative humidity'}))
        if self.background_gas is not None:
            name = self.background_gas
            f = self.background_gas_concentration['f']
            pp = self.background_gas_concentration['pp']
            output.append(f.rename(f'ppv_{name}').assign_attrs({'long_name': f'molar fraction of background gas ({name})'}))
            output.append(pp.rename(f'p_{name}').assign_attrs({'long_name': f'partial pressure of background gas ({name}) (Pa)'}))
        for name in self.absorbers:
            f = self.absorbers[name]['f']
            pp = self.absorbers[name]['pp']
            output.append(f.rename(f'ppv_{name}').assign_attrs({'long_name': f'molar fraction {name}'}))
            output.append(pp.rename(f'p_{name}').assign_attrs({'long_name': f'partial pressure {name} (Pa)'}))
        output.append(self.kappa.rename('kappa').assign_attrs({'long_name': 'total mass absorption coeff (m2/kg)'}))
        output.extend(self.kappas.values())
        output.extend([
            tau.rename('tau'),
            xr.DataArray( self.tau_int[-1],
                          coords=nu.coords, name='tau_s', attrs={'long_name': 'total optical thickness'}),
            xr.DataArray( hr,
                          coords=tau.coords, name='hr', attrs={'long_name': 'Heating rate (K/day/cm-1)'}),
            xr.DataArray( (Fup[1:] + Fup[:-1])/2,
                          coords=tau.coords, name='lw_up', attrs={'long_name': 'upward LW flux (W/m2/cm-1)'}),
            xr.DataArray( (Fdn[1:] + Fdn[:-1])/2,
                          coords=tau.coords, name='lw_dn', attrs={'long_name': 'downward LW flux (W/m2/cm-1)'}),
            xr.DataArray( (Fnet[1:] + Fnet[:-1])/2,
                          coords=tau.coords, name='lw_net', attrs={'long_name': 'net LW flux (W/m2/cm-1)'}),
            xr.DataArray( Fup[0],
                          coords=nu.coords, name='olr', attrs={'long_name': 'outgoing LW radiation (W/m2/cm-1)'}),
            xr.DataArray( Fup_srf[0],
                          coords=nu.coords, name='olr_contrib_srf', attrs={'long_name': 'surface contribution to OLR (W/m2/cm-1)'}),
            xr.DataArray( Fup_atm[0],
                          coords=nu.coords, name='olr_contrib_atm', attrs={'long_name': 'atmospheric contribution to OLR (W/m2/cm-1)'}),
            xr.DataArray( Fup[-1],
                          coords=nu.coords, name='lw_up_srf', attrs={'long_name': 'upward surface LW flux (W/m2/cm-1)'}),
            xr.DataArray( Fdn[-1],
                          coords=nu.coords, name='lw_dn_srf', attrs={'long_name': 'downward surface LW flux (W/m2/cm-1)'}),
            xr.DataArray( Fnet[-1],
                          coords=nu.coords, name='lw_net_srf', attrs={'long_name': 'net surface LW flux (W/m2/cm-1)'}),
            xr.DataArray( self.Tbrightness(nu, Fup[0]).rename('Tb'),
                          coords=nu.coords, name='Tb', attrs={'long_name': 'brightness temperature seen from space (K)'}),
            xr.DataArray( xr.where(-Fdn[-1] > 0, self.Tbrightness(nu, np.maximum(-Fdn[-1], 1.e-32)), np.nan).rename('Tb_srf'),
                          coords=nu.coords, name='Tb_srf', attrs={'long_name': 'brightness temperature seen from surface (K)'})
                   ])
        output = xr.merge(output).drop_attrs(deep=False)

        # add parameter values 
        param_names = ['D', 'line_shape', 'cutoff', 'force_lines_to_grid', 'binning', 'Nbins_gamma', 'Nbins_alpha',
                       'include_mtckd_continuum', 'closure']
        param_values = [f'{D}', f'{line_shape}', f'{cutoff}', f'{force_lines_to_grid}', f'{binning}', f'{Nbins_gamma}', f'{Nbins_alpha}',
                       f'{include_mtckd_continuum}', f'{closure}']
        output = output.assign_attrs(dict(zip(param_names, param_values)))

        # rescale pressure for output
        p = (p/100).assign_attrs({'long_name':'pressure (hPa)'})
        output = output.assign_coords({'p':p})

        return output

    def get_nu_grid(self, nu_min, nu_max, dnu):
        '''
        Returns target wavenumber grid, calculated here to make sure all routines do it consistently
        '''
        nu = np.linspace(nu_min, nu_max, int((nu_max-nu_min)/dnu)+1)
        return xr.DataArray(nu, coords={'nu':nu}, name='nu', attrs={'long_name':'wavenumber (cm-1)'})

    def get_number_fraction_from_specific_humidity(self, q, background_gas, other_gases):
        '''
        Given specific humidity (kg H2O/kg moist gas), the name of the background gas
        and the number fraction of other gases, compute the number fraction of H2O
        '''
        # molecular weight background gas
        m_b = 0.
        if background_gas is not None:
            m_b = phys.gases[background_gas].MolecularWeight

        # total no. fraction and mean mol weight other gases
        m_o = 0
        f_o = 0.
        if len(other_gases) > 0:
            for name in other_gases:
                f_o += other_gases[name]['f']
                m_o += phys.gases[name].MolecularWeight * other_gases[name]['f']

        # molecular weight water vapor
        m_H2O = phys.gases['H2O'].MolecularWeight

        # solve for no. fraction of H2O
        f_H2O = q*(m_o + (1-f_o)*m_b)/(m_H2O - q*(m_H2O - m_b))
        return f_H2O
       
    def Planck(self, nu, T):
        '''
        Compute Planck function (of wavenumber in cm-1)
        Input:
        nu: xarray, wavenumber (cm-1)
        T: scalar or xarray, temperature (K)
        Returns:
        B, xarray, (W/m2/steradian/cm-1)
        '''
        nu = nu*100 # cm-1 -> m-1
        u = phys.h*phys.c*nu/phys.k/T
        u = xr.where(u < 500., u, 500.) # to prevent overflow
        B = 2*phys.h*phys.c**2*nu**3 / (np.exp(u) - 1)
        B = B*100 # W/m2/ster/m-1 -> W/m2/ster/cm-1
        return B.transpose()

    def Tbrightness(self, nu, F):
        '''
        Compute brightness temperature corresponding to
        flux F (W/m2/cm-1) at wavenumber nu (cm-1)
        '''
        if isinstance(nu, np.ndarray):
            nu = xr.DataArray(nu, coords={'nu':nu})
        if isinstance(F, np.ndarray):
            F = xr.DataArray(F, coords={'nu':nu})
        nu = nu*100 # cm-1 -> m-1
        R = F/np.pi/100 # flux W/m2/cm-1 -> radiance W/m2/steradian/m-1
        oneoverTb = phys.k/phys.h/phys.c/nu*np.log(2*phys.h*phys.c**2*nu**3/R + 1)
        return 1/oneoverTb

    def get_partition_sum(self, mol_name, T):
        '''
        Computes partition sum Q for molecule mol_name and temperature T
        T must be float or xarray
        '''
        molec_id = int(self.line_data[mol_name].molec_id[0])
        if isinstance(T, float):
            Q = hapi.partitionSum(molec_id, 1, T)
        else:
             Q = hapi.partitionSum(molec_id, 1, list(T.to_numpy()))
             Q = xr.DataArray(np.array(Q), coords=T.coords)
        return Q

    def format_input(self, p, *args):
        # input -> output:
        # xarray -> xarray
        # list, ndarray -> xarray with p as coord
        # int, float -> float
        # None -> None
        output = []
        if isinstance(p, xr.DataArray):
            pass
        elif isinstance(p, list) or isinstance(p, np.ndarray):
            p = xr.DataArray(p, coords={'p':p}, name='p', attrs={'long_name':'pressure (Pa)'})
        else:
            p = float(p)
        output.append(p)
        for arg in args:
            if (arg is None) or isinstance(arg, xr.DataArray):
                pass
            elif isinstance(arg, list) or isinstance(arg, np.ndarray):
                arg = xr.DataArray(arg, coords={'p':p})
            else:
                arg = float(arg)
            output.append(arg)
        return tuple(output)

    def coarsen(self, ds, dnu, width=10):
        '''
        Coarsen spectral resolution in Dataset ds with wavenumber averages over blocks of given width (cm-1)
        '''
        ds_coarse = ds.coarsen({'nu':int(width/dnu)}, boundary='trim').mean()
        return ds_coarse
