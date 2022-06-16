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

