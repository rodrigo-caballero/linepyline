import numpy as np
import scipy

def two_stream(B, Bs, tau, eps, D):
    '''
    Integrates 2-stream Schwarzschild equations
    tau is optical depth *at layer interfaces*
    B is source function on layer midpoints (W/m2/steradian/cm-1)
    Bs is surface source function (W/m2/steradian/cm-1)
    eps is surface emissivity
    Returns fluxes on interfaces, same units as B
    '''
    # scale for 2 stream
    tau = D * tau
    B = np.pi * B
    Bs = np.pi * Bs

    # compute fluxes (on layer interfaces, defined positive upward)
    Np, Nnu = tau.shape

    # -- downward stream
    Fdown = np.zeros((Np, Nnu))
    for i in range(1, Np):
        trans = np.exp( -np.abs(tau[i] - tau[:i+1]) ) # transmissivity between level i and levels above i
        dtrans = trans[:-1] - trans[1:] 
        Fdown[i,:] = (B[:i]*dtrans).sum(axis=0)

    # -- upward stream, track surface and atmos contributions separately
    Fup_srf = np.zeros((Np, Nnu)) 
    Fup_atm = np.zeros((Np, Nnu))
    Fup_srf[-1] = eps*Bs - (1-eps)*Fdown[-1]
    for i in range(0, Np-1):
        Fup_srf[i] = Fup_srf[-1]*np.exp( -np.abs(tau[i] - tau[-1]) ) # transmissivity between level i and surface
        trans = np.exp( -np.abs(tau[i] - tau[i:]) ) # transmissivity between level i and levels below i
        dtrans = trans[:-1] - trans[1:]
        Fup_atm[i] = (B[i:]*dtrans).sum(axis=0)
    Fup = Fup_srf + Fup_atm

    return Fup_srf, Fup_atm, Fup, Fdown

def heavy_lifting(nu, nu_l, S, gamma, alpha, cutoff, line_shape, remove_plinth, force_lines_to_grid):

    nu_min = nu[0]
    dnu = nu[1] - nu_min

    Ngrid = len(nu)
    Nwin = int(cutoff/dnu) # half window width in gridpoints
    Np, Nlines = S.shape

    # array to hold output
    kappa = np.zeros((Np, Ngrid), dtype='float')
    
    for i in range(Nlines):
        
        # select window (odd no. of points, Nwin gridpoints either side of center)
        i0 = int(np.round( (nu_l[i] - nu_min)/dnu) ) # index of window center = gridpoint closest to line
        i1 = max(0, i0 - Nwin) # start of window
        i2 = min(Ngrid, i0 + Nwin + 1) # end of window
        nu_win = nu[i1:i2] # wavenumbers in window

        # choose where to place the line
        if force_lines_to_grid:                
            nu_center = i0*dnu # line centered at central grid point
        else:
            nu_center = nu_l[i] # line at its real center, can be in between grid points
                                # if lines are thinner than grid resolution dnu, will miss a lot of lines!

        # compute line shape and add to kappa
        for k in range(Np):                                     
            if line_shape == 'lorentz':
                lineshape = lineshape_lorentz(nu_win - nu_center, gamma[k,i]) 
            if line_shape == 'pseudovoigt':
                lineshape = lineshape_pseudovoigt(nu_win - nu_center, alpha[k,i], gamma[k,i]) 
            if line_shape == 'voigt':
                lineshape = lineshape_voigt(nu_win - nu_center, alpha[k,i], gamma[k,i]) 
            if remove_plinth:
                # plinth is defined from lorentzian shape
                lineshape -= lineshape_lorentz(cutoff, gamma[k,i])

            # add contribution of line to kappa
            kappa[k,i1:i2] += S[k,i]*lineshape

    return kappa        

def heavy_lifting_with_binning(nu, nu_l, S, gamma, alpha, Nbins_gamma, Nbins_alpha,
                               cutoff, line_shape, remove_plinth):
    
    nu_min = nu[0]
    dnu = nu[1] - nu_min

    gamma_min = gamma.min()
    alpha_min = alpha.min()

    Ngrid = len(nu)
    Np, Nlines = S.shape

    # binning
    def get_bins(x, Nbins):
        bins = np.linspace(x.min(), x.max()*1.01, Nbins + 1)
        delta = bins[1] - bins[0]
        mbins = (bins[1:]+bins[:-1])/2
        return bins, mbins, delta
    bin_edges_gamma, bin_midpts_gamma, dgamma = get_bins(gamma, Nbins_gamma)
    bin_edges_alpha, bin_midpts_alpha, dalpha = get_bins(alpha, Nbins_alpha)

    # precompute line shapes for each value of gamma, alpha bins
    Nwin = int(cutoff/dnu) # half window width in gridpoints
    nu_win = np.arange(-Nwin, Nwin) * dnu # wns for which to compute lineshape; peak (nu=0) is
                                          # at the start of the right-hand block 
    lineshapes_right = np.zeros((Nbins_gamma, Nbins_alpha, len(nu_win)), dtype='float')
    for ig in range(len(bin_midpts_gamma)):
        for ia in range(len(bin_midpts_alpha)):
            if line_shape == 'lorentz':
                lineshapes_right[ig, ia] = lineshape_lorentz(nu_win, bin_midpts_gamma[ig])
            if line_shape == 'pseudovoigt':
                lineshapes_right[ig, ia] = lineshape_pseudovoigt(nu_win, bin_midpts_alpha[ia], bin_midpts_gamma[ig])
            if line_shape == 'voigt':
                lineshapes_right[ig, ia] = lineshape_voigt(nu_win, bin_midpts_alpha[ia], bin_midpts_gamma[ig])
            if remove_plinth:
                # plinth is defined from lorentzian shape
                lineshapes_right[ig, ia] -= lineshape_lorentz(cutoff, bin_midpts_gamma[ig])
    lineshapes_left = lineshapes_right[:,:,::-1]
    
    # array to hold output
    kappa = np.zeros((Np, Ngrid), dtype='float')
    
    # sum over lines
    for i in range(Nlines):

        # select grid interval where lineshape will be placed
        i0 = int((nu_l[i] - nu_min)/dnu) + 1 # gridpoint to the right of line
                                             # line falls between in [i0-1, i0]
        i1 = max(0, i0 - Nwin) # start 
        i2 = min(Ngrid, i0 + Nwin) # end

        # distance from line to right-hand gridpoint as fraction of grid spacing
        # gives proportion of lineshape that should come from lineshapes_left
        if i0 < 0:
            eps = 1
        if i0 > Ngrid:
            eps = 0
        else:
            eps = (nu[i0] - nu_l[i])/dnu

        for k in range(Np):

            # pick line shape
            ig = int((gamma[k,i] - gamma_min)/dgamma) 
            ia = int((alpha[k,i] - alpha_min)/dalpha) 
            lineshape = eps*lineshapes_left[ig,ia] + (1-eps)*lineshapes_right[ig,ia]

            # catch cases where lineshape window extends beyond ends of grid
            Ninterval = i2 - i1  # no. of lineshape gridpoints on-grid
            if Ninterval < len(nu_win):
                if i1 == 0:
                    # interval at start of grid, take right-hand end of lineshape
                    lineshape = lineshape[len(nu_win) - Ninterval:]
                if i2 == Ngrid:
                    # interval at end of grid, take left-hand end of lineshape
                    lineshape = lineshape[:Ninterval]

            # add contribution of line to kappa
            kappa[k, i1:i2] += S[k, i]*lineshape

    return kappa

"""
Line shape functions 
"""
def lineshape_gaussian(x, alpha):
    """ 
    Return Gaussian line shape at x with HWHM alpha 
    """
    sigma = alpha/np.sqrt(2*np.log(2)) # HWHM -> std dev
    return np.exp(-(x/sigma)**2/2) / np.sqrt(2*np.pi)/sigma

def lineshape_lorentz(x, gamma):
    """ 
    Return Lorentzian line shape at x with HWHM gamma 
    """
    return gamma/np.pi/(x**2 + gamma**2)

def lineshape_pseudovoigt(x, alpha, gamma):
    """
    Return the pseudo-Voigt line shape at x, linear mixture of Lorentzian component HWHM gamma and Gaussian component HWHM alpha.
    Originally proposedd by Thompson, et al. (1987). J. Appl. Cryst. 20, 79-83.
    See also Ida et al. (2000). J. of Appl. Cryst. 33, 1311-1316.
    """
    # effective half-width
    width = (alpha**5 + 2.69269*gamma*alpha**4 + 2.42843*gamma**2*alpha**3
             + 4.47163*gamma**3*alpha**2 + 0.07842*gamma**4*alpha + gamma**5)**(1./5.)
    # proportion of the profile that should be Lorentzian
    eta = 1.36603*(gamma/width) - 0.47719*(gamma/width)**2 + 0.11116*(gamma/width)**3
    return (1 - eta)*lineshape_gaussian(x, width) + eta*lineshape_lorentz(x, width)

def lineshape_voigt(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma and Gaussian component HWHM alpha.
    """
    sigma = alpha/np.sqrt(2*np.log(2))
    #return np.real(scipy.special.wofz((x + 1j*gamma)/(sigma*np.sqrt(2))))/(sigma*np.sqrt(2*np.pi))
    return scipy.special.voigt_profile(x, sigma, gamma)
