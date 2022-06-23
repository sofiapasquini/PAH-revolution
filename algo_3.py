'''
This is the main script in which all pre-processing, processing, 
ML algorithms, and analysis occurs for the algorithm version 3.

Algorithm version 3 is a combination of both agglomerative clustering and random forest
algorithms. In this version, a dissimilarity score is computed for the data using a random
forest classifier (a synthetic dataset is created by sampling from the marginal distributions
of the real data, and the classifier learning correlation between the features). The dissimilarity
matrix is used as a euclidean analog and is passed as input to the agglomerative clustering
algorithm. PCA decomposition reveals the cluster results in PC-space (as in both algorithm
versions 1 and 2).

This script is originally coded for the test Spitzer data, but will
eventually be edited to accomodate the JWST data (perhaps a separate script?).

    In the case of Spitzer data here (NGC2023- from Boersma et al 2016, Zhang et al. 2019),
    each map has both a North and South version, for now I will just code in 
    the script for the North maps, can revisit this concept once code is working.

'''

from processing import *
from spec_build import *
from pre_processing import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS 
import seaborn as sns

##SOFIA- currently working only with the North files, pondering how best to
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

#processing- the algorithm itself

#start by determining the optimal number of clusters to use

clusters_range=[2,3,4,5,6,7,8] #can change to anything you like

#visually inspect the clusters to determine the optimal number of clusters
optimal_clusters_inspect(clusters_range, df)

#prompt the user for the optimal number of clusters going forwards
optimal_n_clusters=int(input("Please input the optimal number of clusters: "))

#create the synthetic dataset
df_synthetic=synthetic_sampler(df)

#create the training and testing datasets

#start by adding a row to the real/synthetic arrays that has the labels
df, df_synthetic=add_labels(df, df_synthetic)

df_whole, X, y=design_stack(df, df_synthetic)

#now split the data into training and testing sets
test_fraction=0.3 #the fraction of the dataset to be used for testing
#set shuffle=True since synthetihe porbsc/real labels are not shuffled in dataset
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_fraction, random_state=0, shuffle=True)

#initialize the classifier
model=RandomForestClassifier()#SOFIA- come back and tune parameters afterwards!

#train and predict!
model.fit(X_train, y_train) #train
y_pred=model.predict(X_test) #test

#compute some metrics to see how the classifier performed
compute_performance(y_pred, y_test, model.classes_)

#compute the similarity matrix
sim_m=similarity_matrix(model, X_train, normalize=True)

#compute the dissimilarity matrix
#SOFIA- is this relevant!?.... keeping it for now...
dis_matrix=dissimilarity_matrix(sim_m)

#now start the MDS
mds=MDS(n_components=2, dissimilarity='precomputed') #using two components as we did in PCA
embedding=mds.fit_transform(dis_matrix)

#pass the embeddings to Agglomerative Clustering
clusterer=AgglomerativeClustering(n_clusters=optimal_n_clusters)
results= clusterer.fit_predict(embedding)

#now visualize the results (embeddings color-coded by cluster labels)
sns.scatterplot(embedding[:,0], embedding[:,1], hue=results)
plt.show()