'''
This module contains all methods related to the processsing of the data
for any three of the algorithms in this project. These methods will
perform actions on the cleaned datasets (steps in the algorithms/pipelines
after the 'cleaning'/pre-processing has taken place). This module also 
includes functions which perform dimensionality reduction, clustering, and
other related visualizations on said data that may not technically 
modify/"process" the original version of the cleaned data.

Coded originally for the Spitzer data cube format, will be edited in accordance
with the JWST data cube format.
'''

import numpy as np
from scipy.linalg import sqrtm
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def synthetic_sampler(df):
    '''
    This is a function which, given an input design matrix, will create a 
    version of the design matrix in which there is no correlation between
    features by sampling from the marginal distributions of the samples
    to create synthetic observations. 

    Inputs: 
        df: 2-D array-like design matrix (following the standard format where
        each row represents an observation and each column represents a feature).

    Ouputs:
        A 2-D array-like of the same shape as the input design matrix. This
        matrix contains the synthetic observations.
    '''

    #calculate the stats on each wavlength in the dataframe
    means=df.mean(axis=0) #the means
    stds=df.std(axis=0) #the standard deviations

    #create the synthetic dataset in the same shape as the original
    df_synthetic=np.zeros((df.shape))#an empty array to hold the synthetic data
    sample_size=df.shape[0]
    for i in range(df.shape[1]): #iterate for each wavelength value
        #create the synthetic sample and fill in the empty array appropriately
        df_synthetic[:,i]=np.random.normal(means[i], stds[i], sample_size)


    return df_synthetic


def add_labels(df, df_synthetic):

    '''
    This is a function that adds a column of labels (either 1's or 0's) to 
    the corresponding arrays and returns the labelled version of the arrays.
    This functionality is intended for arrays which hold data requiring a 
    binary label (for classification purposes).

    Inputs:
        df: a 2-D array-like; the array which holds the data to be labelled "1".

        df_synthetic: a 2-D array-like; the array which holds the data to be
            labelled "0".

    Outputs:
        (df, df_synthetic): a tuple of the labelled versions of the input arrays
            where the first array is the positive class and the second array is
            the negative class.
    '''

    #start by adding a row to the real/synthetic arrays that has the labels
    labels_real=np.ones((df[:,0].shape)) #create the labels as individual columns
    labels_syn=np.zeros((df[:,0].shape))

    #adding the labels to their respective datasets as columns
    df=np.c_[df,labels_real] 
    df_synthetic=np.c_[df_synthetic, labels_syn]


    return df, df_synthetic


def design_stack(df, df_synthetic):
    '''
    This is a function which combines labelled arrays of data from binary classes
    into one main dataframe. The design matrix and label components of the 
    dataframe are also identified and returned along with the whole dataframe.

    Inputs: 
        df: a 2-D array-like; the array holding the labelled data of the positive class.

        df_synthetic: a 2-D array-like; the array holding the labelled data of the
            negative class.

    Outputs: 
    
        (df_whole, X, y): a tuple of the entire labelled dataframe, the design matrix
        only, and the corresponding labels, respectively.

        df_whole: a 2-D array-like; the array holding the labelled data from both
            positive and negative classes concatenated together.
            NOTE: the observations belonging to each class are not shuffled here.

        X: a 2-D array-like; the array holding the design matrix component of
            df_whole (just the data without the column holding the labels).

        y: a 1-D array-like; the column of the labels corresponding to the data
            in the design matrix X (both from df_whole).
    '''

    #combine the real and synthetic datasets into one single array
    df_whole=np.vstack((df, df_synthetic))

    #identify the labels and data in the dataframe
    X=df_whole[:,0:-1] #the data
    y=df_whole[:,-1] #the labels

    return df_whole, X, y


def compute_performance(yhat, y, classes):
    '''
    This function computes multiple statistical metrics which quantify the 
    performance of a given classifier. The output summary statement
    includes values for accuracy, precision, recall, sensitivity and 
    specificity, each rounded to three decimal places.

    Inputs:
        yhat: a 1D array- or list-like where each element is the predictions 
            made by the classifier

        y: a 1D array- or list-like where each element is the true label.

        classes: a 1D array- or list-like where each element is a possible label.
    Outputs:
          A statement summarizing the performance of a classifier based on the
          input predictions and labels. The statement includes values for
          accuracy, precision, recall, sensitivity and specificity, each rounded to 
          three decimal places.
    '''



    # First, get tp, tn, fp, fn
    tp = sum(np.logical_and(yhat == classes[1], y == classes[1]))
    tn = sum(np.logical_and(yhat == classes[0], y == classes[0]))
    fp = sum(np.logical_and(yhat == classes[1], y == classes[0]))
    fn = sum(np.logical_and(yhat == classes[0], y == classes[1]))

    print(f"tp: {tp} tn: {tn} fp: {fp} fn: {fn}")
    
    # Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision
    # "Of the ones I labeled +, how many are actually +?"
    precision = tp / (tp + fp)
    
    # Recall
    # "Of all the + in the data, how many do I correctly label?"
    recall = tp / (tp + fn)    
    
    # Sensitivity
    # "Of all the + in the data, how many do I correctly label?"
    sensitivity = recall
    
    # Specificity
    # "Of all the - in the data, how many do I correctly label?"
    specificity = tn / (fp + tn)
    
    # Print results
    
    print("Accuracy:",round(acc,3),"Recall:",round(recall,3),"Precision:",round(precision,3),
          "Sensitivity:",round(sensitivity,3),"Specificity:",round(specificity,3))


def similarity_matrix(model, X, normalize=True):
    '''
    This is a function that computes a similarity matrix given a model which
    was fit on a specified design matrix. It is assumed that the model used
    is a decision tree ensemble (namely 
    sklearn.ensemble.RandomForestClassifier()).

    Inputs:
        model: the instance of the ensembled decision tree model used in this 
        instance.

        X: an array-like (2-dimensional?) design matrix. In this context
            each row represents an observation and each column represents
            a wavelength value.

        normalize: a keyword that specifies whether or not to normalize the
            distance matrix based on the number of trees in the forest.

    Outputs:
        The computed similarity matrix, a 2-dimensional array-like.
    
    '''
    terminals=model.apply(X)# apply the trees in the forest to X and return the indices of the leaves that each x in X end up in
    #terminals is matrix with number of rows same as X and onr column for each tree
    n_trees=terminals.shape[1]# the number of trees in the forest
    
    sim_matrix=np.zeros((terminals.shape[0], terminals.shape[0])) # create the empty matrix
    
    for i in range(0, n_trees): #iterate through each tree in the forest, fill in the matrix
        a=terminals[:,i] #grab each of the "similarity vectors"
        sim_matrix+=1*np.equal.outer(a,a) #add to the similarity matrix (ie element-wise addition of scores)
        
    if normalize: #is set to True (by default, yes)
        sim_matrix=sim_matrix/n_trees #divide each element by the number of trees in the forest to normalize the scores to be from 0 to 1
        
    return sim_matrix #return the computed similarity matrix


def dissimilarity_matrix(s):
    '''
    This function computes a disimilarity matrix based on an input similarity
    matrix.

    SOFIA- finish this docstring once you actually figure out whether this is
    the kind of matrix that is relevant to the algorithms/project.
    '''


    ##SOFIA- is it d= sqrt(1-s) or just d=1-s???
    
    #first compute the matrix that is 1-s (recall: s is a square matrix)
    d=np.zeros((s.shape))
    for i in range(s.shape[0]): #along the first axis
        for j in range(s.shape[0]): #along the second axis
            d[i,j]=1-s[i,j] # fill in the dissimilarity matrix-each element is 1 minus the element value in s
            
    #SOFIA- commenting this part out for now- only works for positive matrices, returning a matrix with imaginary elements
    #now compute the square root to get the Euclidean distance?...
    #d=sqrtm(s) 
    
    #return the dissimilarity matrix
    return d


def optimal_clusters_inspect(n_clusters, X):
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

        axs.set_title("Average Silhouette score for "+str(n)+" clusters = "+str(silhouette_avg))
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