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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from processing import optimal_clusters_inspect, pca_visual, elbow_plot
from spec_build import *
from pre_processing import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##SOFIA- currently working only with the SOUTH files, pondering how best to
#consolidate the South data files into the dataset

#load in and reshape all wavelength, extinction, spectral maps
spectra, wave= load_spec_wavelength("NGC2023_SPECTRAL_MAP_SOUTH.fits")

#load in and reshape continuum and corresponding wavelentgh arrays
cont, wave_cont=load_continuum("NGC2023_CONTINUUM_MAP_SOUTH.fits")

#load in and reshape the extinction maps and corresponding wavelength array
ext, wave_ext=load_extinction("NGC2023_EXTINCTION_MAPS_SOUTH.fits")

#extinction correct the spectra
ext_corr_spec=extinction_correct(ext, spectra)

#pre-processing

#transform the map(s) from 3 to 2-dimensional array(s and consolidate)
df=df_create(ext_corr_spec)

#apply normalization to the spectra
df=normalize(df)
#uncomment the line below if you want to normalize wrt the 7.7 micro meter flux:
# df=normalize_77(df)

#processing- the algorithm itself

#start by determining the optimal number of clusters to use

clusters_range=[2,3,4,5,6,7,8] #can change to anything you like

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
agglo_cluster_labels=clusterer_1.fit_predict(df_pca)
#calculate the silhouette coefficient for the initial agglomerative clustering
agglo_silhouette=silhouette_score(df_pca, agglo_cluster_labels)

#now compute the centroids of the clusters in the PC space
cent=NearestCentroid()
centroids=cent.fit(df_pca, agglo_cluster_labels).centroids_

#now pass the centroids as initial centroid locations for KMeans clustering
clusterer_2 = KMeans(n_clusters=optimal_n_clusters, random_state=10, init=centroids) #use a random state for reproducibility
kmeans_cluster_labels = clusterer_2.fit_predict(df_pca) #is this correct? The input to the kmeans clustering should be
#calculate the silhouette coefficient for the kmeans clustering
kmeans_silhouette=silhouette_score(df_pca, kmeans_cluster_labels)

#calculate and report the improvement of the silhouette score by the kmeans clustering
silhouette_improved=kmeans_silhouette-agglo_silhouette #note: if negative the score worsened


#now visualize the results
import matplotlib.cm as cm
fig, axs=plt.subplots(1,1)
colors = cm.nipy_spectral(kmeans_cluster_labels.astype(float) / 5)
axs.scatter(df_pca[:, 0], df_pca[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
axs.set_xlabel("First Component Space")
axs.set_ylabel("Second Component Space")
axs.set_title(f"Avg Silhouette Score: {round(kmeans_silhouette,4)}, improved by: {round(silhouette_improved,4)}.")
plt.suptitle("The visualization of the clustered data for "+ str(optimal_n_clusters)+" clusters.")
plt.plot(centroids[:,0], centroids[:,1], "*", color='red') #these are the cluster centroids
plt.show()

#analyze key features of spectra groups/clusters here