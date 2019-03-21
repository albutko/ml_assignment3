
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from timeit import default_timer as timer

from datasets import HiggsBosonDataset, MappingDataset
from utils import *

higgs = HiggsBosonDataset()
mapping = MappingDataset()

def main():
    KMeansExperiment(higgs)

def KMeansExperiment(dataset):
    X, y = dataset.get_train_data()
    if dataset.name == 'higgs':
        kmeans = KMeans(n_clusters = 3, random_state=0)
        bench_clustering(kmeans, 'kmeans', X, y)
        plot_pca_reduced_data(X, kmeans, true_labels=y)
    else:
        pass


if __name__=='__main__':
    main()
