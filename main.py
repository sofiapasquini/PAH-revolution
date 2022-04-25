'''
This is the main script in which all pre-processing, processing, 
ML algorithms, and analysis occurs.

This script is originally coded for the test Spitzer data, but will
eventually be edited to accomodate the JWST data (perhaps a separate script?).

    In the case of Spitzer data here (NGC2023- from Boersma et al 2016, Zhang et al. 2019),
    each map has both a North and South version, for now I will just code in 
    the script for the North maps, can revisit this concept once code is working.

'''
#import relevant packages
from spec_build import *

#load in and reshape all wavelength, extinction, spectral maps
spectra, wave= load_spec_wavelength("NGC2023_SPECTRAL_MAP_NORTH.fits")

#load in and reshape continuum and corresponding wavelentgh arrays
cont, wave_cont=load_continuum("NGC2023_CONTINUUM_MAP_NORTH.fits")

#load in and reshape the extinction maps and corresponding wavelength array
ext, wave_ext=load_extinction("NGC2023_EXTINCTION_MAPS_NORTH.fits")

#extinction correct the spectra
ext_corr_spec=extinction_correct(ext, spectra)