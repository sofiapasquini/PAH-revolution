'''
This module contains all functions related to the clustering algorithms
explored in this project.

There are three algorithms to be explored: basic agglomerative clustering, 
agglomerative clustering followed by kmeans clustering, and agglomerative 
clustering followed by the calculation of a Random Forest score (possibly).

It is assumed that each of these algorithms will be called on the cleaned 
data and as such has been transformed to a 2-Dimensional format (fluxes along)
the 0th axis and wavelength values along the 1st.

The idea is that this module will perform the clustering (processing)
and all visualization/analysis of results can be done using methods of
another module (ie analysis.py).

'''

#import relevant packages
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def optimal_clusters_plot(n_clusters, X):
    '''
    This function helps to determine the optimal number of clusters to use
    for a given data set using silhouette scores. For any given input number
    of clusters, the function performs agglomerative clustering with 
    random_seed=0, calculates a silhouette score for each cluster, and 
    returns a figure with a silhouette plot for each. Visual inspection
    of each of the silhouette plots returned for the given number of clusters
    input will result in determination of the optimal number of clusters.

    Inputs:
        n_clusters: a 1D array- or list-like where each element is the number
                    of clusters to be explored.

        X: 2-D array; the cleaned data set where fluxes are stored along the 0th
            axis and wavelength values are stored along the 1st axis.
    Outputs:
          Figures containing a silhouette plot for each
            number of clusters specified to be explored by 
            the clustering algorithm.
    '''

    for n in n_clusters:
        #get the figure details ready- we are creating a figure of silhouette plots
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        axs.set_xlim([-1, 1])
        axs.set_ylim([0, len(X) + (n + 1) * 10]) #some space between plots

        #initialize the clusterer with n_clusters and a random seed for 
        #reproducibility in case of KMeans
        clusterer=AgglomerativeClustering(n_clusters=n)

        #generate the cluster labels
        cluster_labels=clusterer.fit_predict(X)

        #calculate average silhouette scores for all the samples
        silhouette_avg=silhouette_score(X, cluster_labels)

        #calculate the silhouette scores for each sample
        sample_silhouette_values=silhouette_samples(X, cluster_labels)
        
        #now do the plotting
        y_lower = 10 #for the bottom of the first plot
        for i in range(n):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n)
            axs.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            axs.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        axs.set_title(f"Average Silhouette score for {n} clusters = {silhouette_avg}")
        axs.set_xlabel("The silhouette coefficient values")
        axs.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        axs.axvline(x=silhouette_avg, color="red", linestyle="--")

        axs.set_yticks([])  # Clear the yaxis labels / ticks
        axs.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        plt.suptitle(("Silhouette analysis for Agglomerative clustering on sample data "
                    "with n_clusters = %d" % n),
                    fontsize=14, fontweight='bold')

    # plt.savefig("silhouette_fig.png")
    plt.show()


def pca_visual(X, n_clusters):
    '''
    Performs PCA decomposition and returns a figure of Agglomerative clustering
    color-coded by PCs for the number of clusters specified.

    Inputs:
        X: 2-D array; the cleaned data set where fluxes are stored along the 0th
            axis and wavelength values are stored along the 1st axis.
        
        n_clusters: a 1D array- or list-like where each element is the number
                    of clusters to be explored

    Outputs:
        Figures containing PC plot for each
        number of clusters specified to be explored by 
        the clustering algorithm, color-coded by PCA.

    '''
    pca=PCA(n_components=2)

    #create the plots for each number of clusters specified
    for n in n_clusters:
        fig, axs=plt.subplots(1,1)
        clusterer = AgglomerativeClustering(n_clusters=n)
        cluster_labels = clusterer.fit_predict(X)
        
        #now apply the PCA to the data before plotting
        X_pca=pca.fit_transform(X)
        # print(X_pca.shape)
        
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n)
        axs.scatter(X_pca[:, 0], X_pca[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        axs.set_xlabel("First Component Space")
        axs.set_ylabel("Second Component Space")
        axs.set_title("The visualization of the clustered data for "+ str(n)+" clusters.")

    plt.show()

