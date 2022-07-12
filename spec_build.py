'''
This is a module containing all functions related to reading in and construction 
of spectra from the raw 3-D data cubes.
Coded originally for the Spitzer data cube format, will be edited in accordance
with the JWST data cube format.
'''

from astropy.io import fits
import numpy as np

folder="/Volumes/LaCie/MASTERS/NGC2023_IRS_Boersma_2016/"

def load_spec_wavelength(file_name):
    '''
    Load in wavelength and spectra arrays from the specified file name. 
    Reshape wavelength array from three to one-dimension; the spectral array is re-arranged
    is three-dimensional such that the format is (x, y, lambda).
    The units of the spectrum array are MJy/sr, wavelengths in units of
    micro-meters.
    '''
    #open the file
    hdulist = fits.open(folder+file_name)

    #read in and reshape wavelength array
    wave=hdulist[1].data['wavelength']
    wave=np.reshape(wave, -1)

    #read in and reshape the spectra array
    spectra=hdulist[0].data
    spectra = np.moveaxis(np.swapaxes(spectra, 1, 2), 0, 2) 

    #close the file
    hdulist.close()
    
    return spectra, wave 


def load_continuum(file_name):
    '''
    Load in and re-shape the continuum array as well as the corresponding
    wavelength arrays from the specified file. Both arrays are reshaped to
    three dimensions such that the format is (x, y, lambda).
    The units of the continuum array are MJy/sr, wavelengths in units of
    micro-meters.
    '''

    #open the file
    hdulist=fits.open(folder+file_name)

    #read in and reshape the wavelength arrays
    wave_cont=hdulist[1].data['wavelength']
    wave_cont=np.reshape(wave_cont, [27, 17, 194], order='F')

    #read in and reshape the continuum data
    cont=hdulist[0].data
    cont = np.moveaxis(np.swapaxes(cont, 1, 2), 0, 2)

    #close the file
    hdulist.close()

    return cont, wave_cont


def load_extinction(file_name):
    '''
    Load in and re-shape the extinction map and corresponding wavelength 
    arrays from the specified file. The wavelength array is one-dimensional
    and needs no re-shaping, however the extinction map is re-shapes to three
    dimensions such that the format is (x, y, lambda).
    The visual extinciton (Av) is given in the extinction map as a fraction.
    The units of wavelength are in micro-meters.
    '''

    #open the file
    hdulist=fits.open(folder+file_name)

    #read in the wavelength array
    wave_ext= hdulist[2].data['wavelength']

    #read in and reshape the extinction data
    ext= hdulist[1].data
    ext = np.moveaxis(np.swapaxes(ext, 1, 2), 0, 2)

    #close the file
    hdulist.close()

    return ext, wave_ext


def extinction_correct(ext_array, spec_array):
    '''
    Extinction correct the specified spectrum with a given extinction map.
    Both the input arrays are of the shape (n,m,r) where nxm is the spatial
    coordinates and r is the wavelength axis (for each value each of the 
    arrays there is a single flux or extinction value as appropriate).
    Inputs: 
        ext_array: the extinction map; values are given as fractions
        spec_array: the spectrum array.
    Output: 
        An array of shape (n,m,r) which is the extinction-corrected
        spectral values.
    '''
    
    #divide the spectral values by their corresponding extinction values
    ext_corr=spec_array/ext_array 

    return ext_corr