
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from timeit import default_timer as timer

from datasets import HiggsBosonDataset, MappingDataset
from utils import *

higgs = HiggsBosonDataset()
mapping = MappingDataset()

def main():
    KMeansExperiment(higgs)
    # KMeansExperiment(mapping)

def KMeansExperiment(dataset):
    X, y = dataset.get_train_data()
    if dataset.name == 'higgs':
        kmeans = KMeans(n_clusters = 3, random_state=0)
        # bench_clustering(kmeans, 'kmeans', X, y)
        # plot_pca_reduced_data(X, kmeans, true_labels=y)
        # plot_kmeans_elbow_curve(X)
        plot_kmeans_ami_elbow_curve(X, y)
    else:
        # kmeans = KMeans(n_clusters = 10, random_state=0)
        # bench_clustering(kmeans, 'kmeans', X, y)
        # plot_pca_reduced_data(X, kmeans, true_labels=y)
        plot_kmeans_ami_elbow_curve(X, y)

def GMMExperiment(dataset):
    pass
if __name__=='__main__':
    main()
