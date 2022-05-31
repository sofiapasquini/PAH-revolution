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
from clustering import optimal_clusters_plot, pca_visual
from spec_build import *
from pre_processing import *

##SOFIA- currently working only with the North files, pondering how best to
#consolidate the South data files into the dataset

#load in and reshape all wavelength, extinction, spectral maps
spectra, wave= load_spec_wavelength("NGC2023_SPECTRAL_MAP_NORTH.fits")

#load in and reshape continuum and corresponding wavelentgh arrays
cont, wave_cont=load_continuum("NGC2023_CONTINUUM_MAP_NORTH.fits")

#load in and reshape the extinction maps and corresponding wavelength array
ext, wave_ext=load_extinction("NGC2023_EXTINCTION_MAPS_NORTH.fits")

#extinction correct the spectra
ext_corr_spec=extinction_correct(ext, spectra)

#pre-processing

#transform the map(s) from 3 to 2-dimensional array(s and consolidate)
df=df_create(ext_corr_spec)

#apply z-standardization to the spectra
df=standardize(df)

#processing- the algorithm itself

#start by determining the optimal number of clusters to use

clusters_range=[2,3,4,5,6,7,8] #can change to anything you like

#visually inspect the clusters to determine the optimal number of clusters
optimal_clusters_plot(clusters_range, df)

#prompt the user for the optimal number of clusters going forwards
optimal_n_clusters=input("Please input the optimal number of clusters: ")

#perform a PCA to retreive parameter-space dimensions
pca_visual(df, optimal_n_clusters)# move to above the line before??...
