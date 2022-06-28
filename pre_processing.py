'''
This is a module containing all functions related to the 
pre-processing of the spectra and the construction of the main datasets
from these 'cleaned' spectra.

Coded originally for the Spitzer data cube format, will be edited in accordance
with the JWST data cube format.

'''

import numpy as np

#a function to concatenate the spectra from a map into one large array

def df_create(map):
    '''
    From an input spectral map (should ideally be the cleaned spectra)
    a 2-D array is created such that each row is a single observation, 
    columns corresponding to wavelength value.

    Input:
        3-D spectral map where dimensions correspond to (x, y, lambda)

    Output: 
        2-D array where rows and columns correspond to flux values and
        wavelength locations, respectively.
    
    '''
    #create an empty array to store the data of the appropriate shape
    num_rows=map.shape[0]*map.shape[1]
    num_cols=map.shape[2]
    df=np.zeros(shape=(num_rows, num_cols))
    

    #iterate through the spatial dimensions of the map
    #initialize a pointer to the row in the empty array
    row=0
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            #store the spectra at these coordinates in a row in the array
            df[row]=map[i,j,:]
            #move the pointer to the next row
            row+=1

    #return the array of the cleaned spectra
    return df


def normalize_77(df):
    '''
    This is a function which normalizes the fluxes of spectra in an array to that of the 7.7 micro meter flux.
    Note that this function was written for a dataframe in which the index of the column holding the 7.7 micro meter 
    flux is known and is hard-coded in as the variable idx_77- this value can be altered/changed going forwards/based
    on the use case.

    Inputs:
        df: array-like, nxm where n is the number of spectra and m are the wavelength features.
    
    Outputs:
        an array-like, an nxm array which is the version of the original input matrix which has been 
        normalized to the 7.7 microm feature for all spectra.
    '''
    #find the index of 7.7microm feature
    idx_77=np.where(wave==7.719988) #found this index by manual inspection, not very scientific, can change when/if data changes

    #for each observation, divide each flux value by the 7.7 microm flux
    for i in range(df.shape[0]):
        spectrum=df[i,:] #the entire original spectrum
        normalized_spec=np.divide(spectrum, df[i,idx_77]) #the normalized spectrum
        #replace the original spectrum in the dataframe with the normalized one
        df[i,:]=normalized_spec

    return df
