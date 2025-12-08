###############################################
# Author & Copyright: Konstantinos Kamnitsas
# B1 - Project - 2025
###############################################

import math as m
import numpy as np


def create_data(n_samples):
    """
    Function input:
    n_samples: number of samples to create.
    Function returns:
    X: Two features for each sample.
       It is a matrix of dimensions: n_samples x 2
    class_labels: Class of each sample. Vector of dims: n_samples x 1
                  It has value 1 for Class-1, 2 for Class-2.
    In X and class_labels, row i corresponds to the same sample i
    Note: Returns same number of samples per class (n_samples/2)
    For each class, same number of samples is returned from each of the
    corresponding Gaussians.
    """
    n_classes = 2
    n_samples_per_class = m.floor(n_samples / n_classes)

    # Class 1: 1st col: x1,x2 of mean of Gaussian 1.
    #          2nd col: x1,x2 of mean of Gaussian 2.
    x_mu1 = np.zeros(shape=(2, 2))
    x_mu1[:, 0] = [-0.4, +0.1]
    x_mu1[:, 1] = [+0.1, -1.0]

    # Class 2: 1st col: x1,x2 of mean of Gaussian 1.
    #          2nd col: x1,x2 of mean of Gaussian 2.
    #          3rd col: x1,x2 of mean of Gaussian 3.
    x_mu2 = np.zeros(shape=(2, 3))
    x_mu2[:, 0] = [-0.3, +1.6]
    x_mu2[:, 1] = [+1.3, -0.2]
    x_mu2[:, 2] = [+1.4, +1.6]

    # Diagonal covariance matrices for features of samples from class 1
    # One matrix per Gaussian
    x_var1 = np.zeros(shape=(2, 2, 2))
    x_var1[:, :, 0] = [[0.4, 0.1], [0.1, 0.15]]
    x_var1[:, :, 1] = [[0.2, -0.1], [-0.1, 0.1]]

    # Diagonal covariance matrices for features of samples from class 2
    # One matrix per Gaussian
    x_var2 = np.zeros(shape=(2, 2, 3))
    x_var2[:, :, 0] = [[0.3, 0.0], [0.0, 0.15]]
    x_var2[:, :, 1] = [[0.15, 0.05], [0.05, 0.25]]
    x_var2[:, :, 2] = [[0.15, -0.05], [-0.05, 0.15]]

    # Checks that dimensions are ok.
    if x_mu1.shape[1] != x_var1.shape[2]:
        print("Wrong dimensions for x_mu1!")
        exit()
    if x_mu2.shape[1] != x_var2.shape[2]:
        print("Wrong dimensions for x_mu2!")
        exit()

    # Sampling:
    for c in range(1, n_classes + 1):
        if c == 1:
            x_mu = x_mu1
            x_var = x_var1
        elif c == 2:
            x_mu = x_mu2
            x_var = x_var2
        else:
            print("Not implemented for classes > 2! Error!")
            exit()

        if c == n_classes:  # in case n_classes did not perfectly divide n_samples
            n_samples_per_class = n_samples - n_samples_per_class * (n_classes - 1)

        # ---- Sample X ----
        # Multivariate Normal (MVN):
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html

        n_clusters_from_class = x_mu.shape[1]

        for cluster in range(0, n_clusters_from_class):
            n_samples_from_cluster = m.floor(
                n_samples_per_class / n_clusters_from_class
            )
            if (
                cluster == n_clusters_from_class - 1
            ):  # in case n_clusters_from_class did not perfectly divide n_samples
                n_samples_from_cluster = (
                    n_samples_per_class
                    - n_samples_from_cluster * (n_clusters_from_class - 1)
                )

            # The below returns array dimensions: n_samples x n_features
            X_from_cluster = np.random.multivariate_normal(
                x_mu[:, cluster], x_var[:, :, cluster], n_samples_from_cluster
            )

            if cluster == 0:
                X_for_c = X_from_cluster
            else:
                X_for_c = np.concatenate((X_for_c, X_from_cluster), axis=0)

        # --- Class of each sample (1,2, ....)
        class_labels_for_c = np.ones(shape=(n_samples_per_class)) * c

        # Concat samples of all classes along 1st dimension (rows of samples):
        if c == 1:
            X = X_for_c
            class_labels = class_labels_for_c
        else:
            X = np.concatenate((X, X_for_c), axis=0)
            class_labels = np.concatenate((class_labels, class_labels_for_c), axis=0)

    # Petrube order of the data samples, so that the order of the two
    # classes is random, rather than 1st class first, 2nd class afterwards.
    # ...
    new_order = np.random.permutation(n_samples)
    X = X[new_order, :]
    class_labels = class_labels[new_order]

    print(
        "Create data function finished. " + "Returning X of size: [",
        X.shape,
        "] and " + "class_labels of size: [",
        class_labels.shape,
        "].",
    )

    return [X, class_labels]
