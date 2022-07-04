'''
This script is a mock/trial of the processing/analysis of the NGC 2023 
spectra that takes place in Boersma 2016. The purpose is to re-create the
results of his clustering so that I know my algorithms are on the right track.

The rough premise is simply normalization to the 7.7 line, k-means clustering with
4 clusters, normalization to the peak flux value for visualization purposes.
'''

#import relevant packages
from processing import *
from spec_build import *
from pre_processing import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##SOFIA- currently working only with the South files, pondering how best to
#consolidate the South data files into the dataset

#load in and reshape all wavelength, extinction, spectral maps
spectra, wave= load_spec_wavelength("NGC2023_SPECTRAL_MAP_SOUTH.fits")

#load in and reshape continuum and corresponding wavelentgh arrays
cont, wave_cont=load_continuum("NGC2023_CONTINUUM_MAP_SOUTH.fits")

#load in and reshape the extinction maps and corresponding wavelength array
ext, wave_ext=load_extinction("NGC2023_EXTINCTION_MAPS_SOUTH.fits")

#extinction correct the spectra
##SOFIA-comment this line out when performing clustering on spectra that have
#NOT been extinction corrected
# ext_corr_spec=extinction_correct(ext, spectra)

#pre-processing

#transform the map(s) from 3 to 2-dimensional array(s and consolidate)
# df=df_create(ext_corr_spec)
df=df_create(spectra) #the analysis will be done (for now) on spectra NOT ext-corrected

#apply normalization to the spectra
# df=normalize(df)
#uncomment the line below if you want to normalize wrt the 7.7 micro meter flux:
df=normalize_77(df)

optimal_n_clusters=4

#apply the agglomerative clustering algorithm with optimal number of clusters
clusterer=KMeans(n_clusters=optimal_n_clusters, compute_distances=True) #SOFIA- experiment/finalize parameter settings
cluster_labels=clusterer.fit_predict(df)

#analyze key features of spectra groups/clusters here

#reshape the label matrix from 1-D back to 2-D to match the spectrum matrix
label_matrix=label_reshape(cluster_labels, spectra)
spec_list=[] # an empty list to hold all of the averaged spectra
for i in range(optimal_n_clusters):
    # spec_list.append(avg_label(i,spectra, label_matrix))
    avg_spec=avg_label(i, spectra, label_matrix)
    #normalize the spectra to the peak wavelength
    norm_spec=normalize_peak(avg_spec)
    plt.plot(wave, norm_spec, label=str(i))
#now plot the averaged spectra together
plt.xlabel("Wavelength [$\mu$m]")
plt.ylabel("Normalized Flux")
plt.legend()
plt.show()


##SOFIA- insert code here that creates a figure which is the image of NGC 2023/the FOV overlain
#with the cluster zones (ex Fig 3 in B2014)

