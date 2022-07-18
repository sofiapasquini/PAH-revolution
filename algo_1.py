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
from processing import *
from spec_build import *
from pre_processing import *
from sklearn.cluster import AgglomerativeClustering, KMeans
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

#subtract continuum and extinction correct the spectra



##SOFIA-comment this line out when performing clustering on spectra that have
#NOT been extinction corrected
# ext_corr_spec=extinction_correct(ext, spectra)

#pre-processing

#transform the map(s) from 3 to 2-dimensional array(s and consolidate)
# df=df_create(ext_corr_spec)

df=df_create(spectra) #the analysis will be done (for now) on spectra NOT ext-corrected

#apply normalization to the spectra (unit norm per spectra, not to 7.7 line as previously thought)
df=normalize(df)
#uncomment the line below if you want to normalize wrt the 7.7 micro meter flux:
# df=normalize_77(df)

#load in and adjust the pixel mask:
map_file= "NGC2023_ZONES_MAP_SOUTH.fits"
hdulist = fits.open(folder+map_file)
map=hdulist[0].data

#reshape the axes to (x,y) coordinates and to 1D in order to mask out appropriate rows
map=np.swapaxes(map, 0,1)
map_1d=np.reshape(map, (map.shape[0]*map.shape[1],), order='c')
to_drop=np.where(map_1d==0)[0]

#remove the masked spectra from the data:
df=mask_clean(df, map_1d)

#processing- the algorithm itself

#start by determining the optimal number of clusters to use

clusters_range=[2,3,4,5,6,7,8] #can change to anything you like

#create an elbow plot to visualize the difference between the quality of clustering for each number of clusters
elbow_plot(cluster_range=clusters_range, data=df)

#visually inspect the clusters to determine the optimal number of clusters
optimal_clusters_inspect(clusters_range, df)

#prompt the user for the optimal number of clusters going forwards
optimal_n_clusters=int(input("Please input the optimal number of clusters: "))

#deploy agglomerative clustering with optimal number of clusters
##SOFIA- consider re-coding this in a method in the clustering module?

#first perform PCA to retrieve parameter space dimensions
pca = PCA(n_components = 2) # use 2 from Ameek's paper- could also experiment with others...
df_pca = pca.fit_transform(df)

#apply the agglomerative clustering algorithm with optimal number of clusters
clusterer=AgglomerativeClustering(n_clusters=optimal_n_clusters, compute_distances=True) #SOFIA- experiment/finalize parameter settings
cluster_labels=clusterer.fit_predict(df_pca)

#now plot the results
#plot in pc space, color-code by cluster
for i in range(optimal_n_clusters):
    color = cm.nipy_spectral(float(i) / optimal_n_clusters)
    plt.scatter(df_pca[cluster_labels==i, 0], df_pca[cluster_labels==i, 1], 
                label='Cluster %i' % (i+1))

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA-transformed plot for %i clusters' % optimal_n_clusters)
plt.legend()
plt.show()


#now lets try for a dendrogram visual
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(clusterer, truncate_mode="none")
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#analyze key features of spectra groups/clusters here
print(cluster_labels.shape)
#reshape the label matrix from 1-D back to 2-D to match the spectrum matrix
# label_matrix=label_reshape(cluster_labels, spectra)
spec_list=[] # an empty list to hold all of the averaged spectra
for i in range(optimal_n_clusters):
    # spec_list.append(avg_label(i,spectra, label_matrix))
    # avg_spec=avg_label(i, spectra, label_matrix)
    avg_spec=avg_label(i,df, cluster_labels)
    #normalize the spectra to the peak wavelength
    norm_spec=normalize_peak(avg_spec)
    plt.plot(wave, norm_spec, label=str(i))
#now plot the averaged spectra together
plt.xlabel("Wavelength [$\mu$m]")
plt.ylabel("Normalized Flux")
plt.legend()
plt.show()
