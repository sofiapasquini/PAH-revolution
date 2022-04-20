#import some relevant packages
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score #(?)
import matplotlib.pyplot as plt
from astropy.io import fits

#this will be the main script in which all pre-processing, processing, ML algorithms, and analysis will occur.