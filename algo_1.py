'''
This is the main script in which all pre-processing, processing, 
ML algorithms, and analysis occurs for the base algorithm version.

The base algorithm version is that which is solely agglomerative clustering with PCA
decomposition for visualization.

This script is originally coded for the test Spitzer data, but will
eventually be edited to accomodate the JWST data (perhaps a separate script?).

    In the case of Spitzer data here (NGC2023- from Boersma et al 2016, Zhang et al. 2019),
    each map has both a North and South version, for now I will just code in 
    the script for the North maps, can revisit this concept once code is working.

'''
#import relevant packages
from tkinter import N
from clustering import optimal_clusters_inspect, pca_visual
from spec_build import *
from pre_processing import *
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
optimal_clusters_inspect(clusters_range, df)

#prompt the user for the optimal number of clusters going forwards
optimal_n_clusters=input("Please input the optimal number of clusters: ")

#deploy agglomerative clustering with optimal number of clusters
##SOFIA- consider re-coding this in a method in the clustering module?

#first perform PCA to retrieve parameter space dimensions
pca = PCA(n_components = 2) # use 2 from Ameek's paper- could also experiment with others...
df_pca = pca.fit_transform(df)

#apply the agglomerative clustering algorithm with optimal number of clusters
clusterer=AgglomerativeClustering(n_clusters=optimal_n_clusters) #SOFIA- experiment/finalize parameter settings
cluster_labels=clusterer.fit_predict(df_pca)

#now plot the results
#plot in pc space, color-code by cluster
for i in range(optimal_clusters):
    color = cm.nipy_spectral(float(i) / optimal_clusters)
    plt.scatter(df_pca[cluster_labels==i, 0], df_pca[cluster_labels==i, 1], 
                label='Cluster %i' % (i+1))

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA-transformed plot for %i clusters' % optimal_n_clusters)
plt.legend()
plt.show()

#analyze key features of spectra groups/clusters here