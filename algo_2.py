'''
This is the main script in which all pre-processing, processing, 
ML algorithms, and analysis occurs for the algorithm version 2.

Algorithm version 2 is the base agglomerative clustering algorithm with the output
passed as starting positions to a k-means clustering algorithm.

This script is originally coded for the test Spitzer data, but will
eventually be edited to accomodate the JWST data (perhaps a separate script?).

    In the case of Spitzer data here (NGC2023- from Boersma et al 2016, Zhang et al. 2019),
    each map has both a North and South version, for now I will just code in 
    the script for the North maps, can revisit this concept once code is working.

'''
#import relevant packages
from turtle import shape
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from processing import optimal_clusters_inspect, pca_visual, elbow_plot, avg_label, normalize_peak
from spec_build import *
from pre_processing import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##SOFIA- currently working only with the SOUTH files, pondering how best to
#consolidate the South data files into the dataset

# folder="/Volumes/LaCie/MASTERS/NGC2023_IRS_Boersma_2016/"
# file_name="NGC2023_SPECTRAL_MAP_SOUTH.fits"
folder="/Volumes/LaCie/MASTERS/NGC7469_MIRI/"
file_name= "MRS_stitched_allchannels.fits"

#load in and reshape all wavelength, extinction, spectral maps
spectra, wave= load_spec_wavelength(folder+file_name,1,2)

#load in and reshape continuum and corresponding wavelentgh arrays
# cont, wave_cont=load_continuum("NGC2023_CONTINUUM_MAP_SOUTH.fits")

#load in and reshape the extinction maps and corresponding wavelength array
# ext, wave_ext=load_extinction("NGC2023_EXTINCTION_MAPS_SOUTH.fits")

#extinction correct the spectra
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

##SOFIA- is this not the same as map_1d, map_2d=spec_build.load_map(folder+map_file)?
# # load in and adjust the pixel mask:
# map_file= "NGC2023_ZONES_MAP_SOUTH.fits"
# hdulist = fits.open(folder+map_file)
# map=hdulist[0].data

# #reshape the axes to (x,y) coordinates and to 1D in order to mask out appropriate rows
# map=np.swapaxes(map, 0,1)
# map_1d=np.reshape(map, (map.shape[0]*map.shape[1],), order='c')
# to_drop=np.where(map_1d==0)[0]

# #remove the masked spectra from the data:
# df=mask_clean(df, map_1d)


#processing- the algorithm itself

#start by determining the optimal number of clusters to use

# clusters_range=[2,3,4,5,6,7,8] #can change to anything you like
clusters_range=[5,6,7,8]

#create an elbow plot to visualize the difference between the quality of clustering for each number of clusters
elbow_plot(cluster_range=clusters_range, data=df)

#visually inspect the clusters to determine the optimal number of clusters
optimal_clusters_inspect(clusters_range, df)

#prompt the user for the optimal number of clusters going forwards
optimal_n_clusters=int(input("Please input the optimal number of clusters: "))

#sofia add in the two clustering algorithms here (see the Algo Pseudo Code nb)

#first perform PCA to retrieve parameter space dimensions
pca = PCA(n_components = 2) # use 2 from Ameek's paper- could also experiment with others...
df_pca = pca.fit_transform(df)

#start with the agglomerative clustering
clusterer_1=AgglomerativeClustering(n_clusters=optimal_n_clusters, compute_distances=True)
agglo_cluster_labels=clusterer_1.fit_predict(df) #fit the algo on the original data- not PCA transformed!?
#calculate the silhouette coefficient for the initial agglomerative clustering
agglo_silhouette=silhouette_score(df, agglo_cluster_labels)

#now compute the centroids of the clusters in the PC space
##SOFIA- isnt it supposed to be calculated in the original space? but how would that work...
cent=NearestCentroid()
centroids=cent.fit(df_pca, agglo_cluster_labels).centroids_

#now pass the centroids as initial centroid locations for KMeans clustering
clusterer_2 = KMeans(n_clusters=optimal_n_clusters, random_state=10, init=centroids) #use a random state for reproducibility
kmeans_cluster_labels = clusterer_2.fit_predict(df_pca) #is this correct? The input to the kmeans clustering should be
#calculate the silhouette coefficient for the kmeans clustering
kmeans_silhouette=silhouette_score(df_pca, kmeans_cluster_labels)

#calculate and report the improvement of the silhouette score by the kmeans clustering
silhouette_improved=kmeans_silhouette-agglo_silhouette #note: if negative the score worsened


# #now visualize the results
# import matplotlib.cm as cm
# fig, axs=plt.subplots(1,1)
# colors = cm.nipy_spectral(kmeans_cluster_labels.astype(float) / 5)
# axs.scatter(df_pca[:, 0], df_pca[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')
# axs.set_xlabel("First Component Space")
# axs.set_ylabel("Second Component Space")
# axs.set_title(f"Avg Silhouette Score: {round(kmeans_silhouette,4)}, improved by: {round(silhouette_improved,4)}.")
# plt.suptitle("The visualization of the clustered data for "+ str(optimal_n_clusters)+" clusters.")
# plt.plot(centroids[:,0], centroids[:,1], "*", color='red') #these are the cluster centroids
# plt.show()

#analyze key features of spectra groups/clusters here

#lets see the labels of the spectra in the spatial coordinate system
#reshape the label matrix back to 2D and plot the colormap
kmeans_cluster_labels_2d=np.reshape(kmeans_cluster_labels, (spectra.shape[0], spectra.shape[1]), order='c')
plt.imshow(kmeans_cluster_labels_2d, cmap='viridis')
plt.colorbar()
plt.title("Cluster Zones from Cluster Results")
# plt.savefig("/Volumes/LaCie/MASTERS/NGC7469_MIRI/Test_Results/agglomerative_results/agglomerative_cluster_results.png")
plt.show()

#reshape the label matrix from 1-D back to 2-D to match the spectrum matrix
print(kmeans_cluster_labels.shape)
# label_matrix=label_reshape(kmeans_cluster_labels, spectra)
spec_list=[] # an empty list to hold all of the averaged spectra
for i in range(optimal_n_clusters):
    # spec_list.append(avg_label(i,spectra, label_matrix))
    avg_spec=avg_label(i, df, kmeans_cluster_labels)
    #normalize the spectra to the peak wavelength
    norm_spec=normalize_peak(avg_spec)
    plt.plot(wave, norm_spec, label=str(i))
#now plot the averaged spectra together
plt.xlabel("Wavelength [$\mu$m]")
plt.ylabel("Flux [MJy/sr]")
plt.legend()
plt.show()