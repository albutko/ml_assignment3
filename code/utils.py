from itertools import cycle
import itertools
from sklearn.model_selection import learning_curve, train_test_split, ParameterGrid, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,f1_score, recall_score, auc
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy import interp
from timeit import default_timer as timer
import csv



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns


def bench_kmeans(estimator, name, data, labels):
    t0 = timer()
    pred_labels = estimator.predict(data)
    print(('Estimator:%-9s\nintertia\t%i\nHomogeneity\t%.3f\nAMI\t\t%.3f')
    %(name, estimator.inertia_,
    metrics.homogeneity_score(labels, pred_labels),
    metrics.adjusted_mutual_info_score(labels,  pred_labels)))

def bench_em(estimator, name, data, labels):
    t0 = timer()
    pred_labels = estimator.predict(data)
    print(('Estimator:%-9s\nHomogeneity\t%.3f\nAMI\t\t%.3f')
    %(name,
    metrics.homogeneity_score(labels, pred_labels),
    metrics.adjusted_mutual_info_score(labels,  pred_labels)))

def plot_pca_reduced_data_em(data, clustering_algorithm, true_labels, label_dict, n_components=2, legend=True):
    ellipse_cat = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    colors_dict = {'a':'blue','b':'orange','c':'green','d':'red','e':'purple','f':'brown','g':'pink','h':'teal','i':'gray','j':'khaki','k':'black','l':'slategrey','m':'fuchsia','n':'lime','o':'indigo'}
    colors = ['blue','orange','green','red','purple','brown','pink','teal','gray','khaki','black','slategrey','fuchsia','lime','indigo']
    plt.clf()
    fig, ax = plt.subplots()
    if true_labels is None:
        true_labels = np.zeros_like(data.shape[0])

    reduced_data = PCA(n_components=2).fit_transform(data)
    clustering_algorithm.fit(reduced_data)
    Y_ = clustering_algorithm.predict(reduced_data)

    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

    labels = []
    ellipse_labels = []

    for idx, inst in enumerate(data):
        labels.append(label_dict[true_labels[idx]])
        ellipse_labels.append(ellipse_cat[Y_[idx]])

    d = {'x':reduced_data[:, 0],'y':reduced_data[:, 1],'category':labels, 'ellipse':ellipse_labels}
    sns.scatterplot(x='x', y='y', style='category', hue='ellipse', palette=colors_dict, data=d, alpha=.6, legend='brief')
    for i, (mean, cov) in enumerate(zip(clustering_algorithm.means_, clustering_algorithm.covariances_)):
        v, w = np.linalg.eigh(cov)
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color = colors[i])
        ell.set_alpha(.5)
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)

    plt.title('GMM clustering on PCA-reduced data\n'
              'Centroids are marked with black crosses')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    if not legend:
        ax.legend().remove()


    plt.show()

def plot_pca_reduced_data_kmeans(data, clustering_algorithm, true_labels, label_dict, n_components=2):
    plt.clf()

    if true_labels is None:
        true_labels = np.zeros_like(data.shape[0])

    reduced_data = PCA(n_components=2).fit_transform(data)
    clustering_algorithm.fit(reduced_data)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = clustering_algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    labels = []
    for cat in true_labels:
        labels.append(label_dict[cat])

    d = {'x':reduced_data[:, 0],'y':reduced_data[:, 1],'category':labels}
    sns.scatterplot(x='x', y='y', hue = 'category', data=d, alpha=.6)

    # Plot the centroids as a white X
    centroids = clustering_algorithm.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=5,
                color='black', zorder=10)
    plt.title('K-means clustering on PCA-reduced data\n'
              'Centroids are marked with black crosses')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()


def plot_kmeans_intertia_elbow_curve(data, k=20, true_labels=None):
    plt.clf()
    ks = np.arange(0,20)+1
    results = []

    for k in ks:
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(data)
        results.append(kmeans.inertia_)

    plt.plot(ks, results)
    plt.show()

def plot_kmeans_elbow_curve_train_test(train_data, test_data, train_labels, test_labels, k=20, score='hg'):
    plt.clf()
    ks = np.arange(0,20)+1
    train_homog = []
    train_ami = []
    test_homog = []
    test_ami = []
    fit_times = []

    for k in ks:
        kmeans = KMeans(n_clusters=k)
        start = timer()
        train_pred_labels = kmeans.fit_predict(train_data)
        fit_times.append(timer()-start)

        test_pred_labels = kmeans.predict(test_data)
        train_homog.append(metrics.homogeneity_score(train_labels,  kmeans.labels_))
        test_homog.append(metrics.homogeneity_score(test_labels,  test_pred_labels))
        train_ami.append(metrics.adjusted_mutual_info_score(train_labels,  kmeans.labels_))
        test_ami.append(metrics.adjusted_mutual_info_score(test_labels,  test_pred_labels))

    plt.title("Homogeneity and Adjusted Mutual Information Score vs Clusters")
    plt.plot(ks, train_homog, linestyle= '--', c='red', label='Train Homogeneity')
    plt.plot(ks, test_homog, linestyle= '--', c='blue', label='Test Homogeneity')
    plt.plot(ks, train_ami, linestyle= '-', c='red', label='Train AMI')
    plt.plot(ks, test_ami, linestyle= '-', c='blue', label='Test AMI')
    plt.legend(loc="upper left")
    plt.show()

    return (ks, fit_times)

def plot_fit_time_train_size(data, estimator):
    plt.clf()
    plt.figure()
    percs = np.linspace(0.1, .99, 10)
    times = []
    sizes = []
    for perc in percs:
        _,X_t,_,y_t = train_test_split(data, np.zeros_like(data), test_size=perc)
        start = timer()
        estimator.fit(X_t)
        times.append(timer()-start)
        sizes.append(X_t.shape[0])

    plt.title("Fit Time by Training Size")
    plt.plot(sizes, times, )
    plt.show()

def plot_em_elbow_curve_train_test(train_data, test_data, train_labels, test_labels, k=20, score='hg'):
    plt.clf()
    ks = np.arange(0,20)+1
    train_homog = []
    train_ami = []
    test_homog = []
    test_ami = []
    fit_times = []

    for k in ks:
        gmm = GaussianMixture(n_components=k)
        start = timer()
        train_pred_labels = gmm.fit_predict(train_data)
        fit_times.append(timer()-start)

        test_pred_labels = gmm.predict(test_data)
        train_homog.append(metrics.homogeneity_score(train_labels,  gmm.predict(train_data)))
        test_homog.append(metrics.homogeneity_score(test_labels,  test_pred_labels))
        train_ami.append(metrics.adjusted_mutual_info_score(train_labels,  gmm.predict(train_data)))
        test_ami.append(metrics.adjusted_mutual_info_score(test_labels,  test_pred_labels))

    plt.title("Homogeneity and Adjusted Mutual Information Score vs Clusters")
    plt.plot(ks, train_homog, linestyle= '--', c='red', label='Train Homogeneity')
    plt.plot(ks, test_homog, linestyle= '--', c='blue', label='Test Homogeneity')
    plt.plot(ks, train_ami, linestyle= '-', c='red', label='Train AMI')
    plt.plot(ks, test_ami, linestyle= '-', c='blue', label='Test AMI')
    plt.legend(loc="upper left")
    plt.show()

    return (ks, fit_times)


def calculate_reprojection_error(data, projector, random=False):
    projected_data = projector.fit_transform(data)
    if random:
        inv_project_matrix = np.linalg.pinv(projector.components_)
        reconstructed_data = projected_data.dot(inv_project_matrix.T)

    else:
        reconstructed_data = projector.inverse_transform(projected_data)

    mse = np.sum(np.sqrt(np.sum((data - reconstructed_data)**2, axis = 1)))
    return mse
