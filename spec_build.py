'''
This is a module containing all functions related to reading in and construction 
of spectra from the raw 3-D data cubes.
Coded originally for the Spitzer data cube format, will be edited in accordance
with the JWST data cube format.
'''

from astropy.io import fits
import numpy as np



def load_spec_wavelength(file_name, sci_ext, wave_ext):
    '''
    Load in wavelength and spectra arrays from the specified file name. 
    Reshape wavelength array from three to one-dimension; the spectral array is re-arranged
    is three-dimensional such that the format is (x, y, lambda).
    The units of the spectrum array are MJy/sr, wavelengths in units of
    micro-meters.

    Inputs:
        file_name: a string, the full path to the input fits file.

        sci_ext: an int, the number of the science extension in the input fits file.
            Note: for JWST data this is the second extension (index 1).

        wave_ext: an int, the number of the extension holding the wavelength data in the
            input fits file. Note: for JWST data this is the third extension (index 2).
    '''
    #open the file
    hdulist = fits.open(file_name)

    #read in and reshape wavelength array
    wave=hdulist[wave_ext].data
    #SOFIA- if working with Spitzer data might have to use this line:
    # wave=hdulist[wave_ext].data['wavelength']

    #spitzer wavelengths have to be reshaped, not JWST
    if len(wave.shape)!=1:
        wave=np.reshape(wave, -1) 

    #read in and reshape the spectra array
    spectra=hdulist[sci_ext].data
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



def continuum_subtract(spec, cont):
    '''
    Subtract the continuum emission component from the specified spectrum with a 
    given continuum map. The input arrays are of the shape (n,m,r) where nxm is the
    spatial dimension and r is the wavelength dimension.

    Inputs:
        spec: the spectrum array (an array-like)
        cont: the continuum array (an array-like)

    Outputs:
        An array of shape nxmxr which is the spectrum which has had continuum-subtraction
        applied.
    '''
    #subtract the continuum flux by the spectral flux values at each wavelength
    cont_sub_spec=np.subtract(spec,cont)

    return cont_sub_spec


def load_map(map_file):
    '''
    This is a function which returns 2 array-like objects containing the mask map
    for any kind of array; extracts the mask from the input fits file and returns the
    mask as both a 2-D and 1-D array. The input coordinates of the mask should correspond to a 
    spatial dimensionality (ie nxm-> x,y).

    Inputs:
        map_file- a string, the full path to the fits file containing the map information.

    Outputs:
        a tuple of array-like objects holding mask values;
            the first element is a 1-D array of length n*m and the second element is
            a 2-D array of shape nxm.

    '''
    #load in the map
    hdulist = fits.open(map_file)

    #grab the 2-D data file (an array-like)
    map_2d=hdulist[0].data 

    #be sure the axes are in the right order (we want x,y)
    map_2d=np.swapaxes(map_2d, 0,1)

    #transform to create the 1-D data file 
    map_1d=np.reshape(map_2d, (map_2d.shape[0]*map_2d.shape[1],), order='c')


    return map_1d, map_2d