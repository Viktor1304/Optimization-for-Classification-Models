###############################################
# Author & Copyright: Konstantinos Kamnitsas
# B1 - Project - 2025
###############################################

from create_data import create_data
import numpy as np
import matplotlib.pyplot as plt  # For plotting


def sigmoid(X, theta):
    z = X @ theta
    return 1.0 / (1.0 + np.exp(-z))


def log_grad(X_train, y_train, theta):
    """
    Vectorized gradient for logistic regression.
    X_train: (N, d)
    y_train: (N,)
    theta:   (d, 1)
    Returns:
        gradient: (d, 1)
    """

    y_pred = sigmoid(X_train, theta)
    error = y_pred - y_train.reshape(-1, 1)
    gradient = X_train.T @ error
    return gradient


def grad_descent(X_train, y_train, learning_rate, grad, iters_total):
    theta = np.zeros((X_train.shape[1], 1))

    N = X_train.shape[0]

    for _ in range(iters_total):
        theta -= learning_rate * grad(X_train, y_train, theta) / N

    return theta


def mean_logloss(X, y_real, theta):
    y_pred = sigmoid(X, theta).reshape(-1)  # (N,)
    y_real = y_real.reshape(-1)

    eps = 1e-12
    loss = -(y_real * np.log(y_pred + eps) + (1 - y_real) * np.log(1 - y_pred + eps))

    return np.mean(loss)


def log_regr(X, theta):
    return sigmoid(X, theta).reshape(-1)


def classif_error(y_real, y_pred):
    """
    y_real: continuous outputs (probabilities)
    y_pred: class labels (0/1)
    """

    y_hat = np.round(y_real).astype(int)
    y_pred = np.array(y_pred).astype(int)

    return 1.0 - np.mean(y_hat == y_pred)


def create_features_for_poly(X, degree):
    x1 = X[:, 0]
    x2 = X[:, 1]

    features = []

    for d in range(degree, -1, -1):
        for i in range(d, -1, -1):
            j = d - i
            features.append((x1**i) * (x2**j))

    return np.vstack(features).T


# --------- Helper Plotting Function ----------------


def plot_data(x, class_labels):
    """
    Plots the data returned from the create_data() function.
    x: Matrix of dimensions number_of_samples x number_of_features.
       This should NOT include the concatenated 1 for the bias.
    class_labels: Vector of dimensions number_of_samples.
                  Expects values class_labels={1,2} . Not the y={0,1}
    """
    # Plot the points
    size_markers = 20

    fig, ax = plt.subplots()
    # Class-1 is Red.
    ax.scatter(
        x[class_labels == 1, 0],
        x[class_labels == 1, 1],
        s=size_markers,
        c="red",
        edgecolors="black",
        linewidth=1.0,
    )
    # Class-2 is Green
    ax.scatter(
        x[class_labels == 2, 0],
        x[class_labels == 2, 1],
        s=size_markers,
        c="green",
        edgecolors="black",
        linewidth=1.0,
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim([-2.0, 3.0])
    ax.set_ylim([-2.0, 3.0])
    ax.legend("class 1", "class 2")
    ax.grid(True)

    plt.show()


def plot_data_line(x, class_labels, theta, degree):
    fig, ax = plt.subplots()

    # --- Plot data points ---
    ax.scatter(
        x[class_labels == 1, 0],
        x[class_labels == 1, 1],
        s=20,
        c="red",
        edgecolors="black",
        label="class 1",
    )
    ax.scatter(
        x[class_labels == 2, 0],
        x[class_labels == 2, 1],
        s=20,
        c="green",
        edgecolors="black",
        label="class 2",
    )

    x1_vals = np.linspace(-2, 3, 300)
    x2_vals = np.linspace(-2, 3, 300)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    grid = np.column_stack([X1.ravel(), X2.ravel()])

    Phi = create_features_for_poly(grid, degree)

    Z = Phi @ theta

    Z = Z.reshape(X1.shape)

    ax.contour(X1, X2, Z, levels=[0], linewidths=2, colors="blue")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim([-2.0, 3.0])
    ax.set_ylim([-2.0, 3.0])
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    # Hyper-parameters:
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    nmb_iters: list[int] = [100, 500, 1000, 10000]
    degree_poly = 3

    param_count: dict[tuple[float, int], int] = {}

    for _ in range(20):
        # Create training data
        n_samples_train = 400
        [X_train, class_labels_train] = create_data(n_samples_train)
        X_train_hat = np.concatenate(
            (X_train, np.ones(shape=(n_samples_train, 1))), axis=1
        )
        # Change class labels ={1,2} to values for y={0,1} respectively.
        y_train = (class_labels_train == 1) * 0 + (class_labels_train == 2) * 1
        X_train_poly = create_features_for_poly(X_train, degree_poly)

        # Create validation data
        n_samples_val = 4000
        [X_val, class_labels_val] = create_data(n_samples_val)
        X_val_hat = np.concatenate((X_val, np.ones(shape=(n_samples_val, 1))), axis=1)
        # Change class labels ={1,2} to values for y={0,1} respectively.
        y_val = (class_labels_val == 1) * 0 + (class_labels_val == 2) * 1
        X_val_poly = create_features_for_poly(X_val, degree_poly)

        optimal_theta = []
        min_error = 10.0
        optimal_hyper = (1, 1)
        # Optimize - Logistic Regression - Gradient Descent
        for lr in learning_rates:
            for gd_iters in nmb_iters:
                theta_opt = grad_descent(X_train_poly, y_train, lr, log_grad, gd_iters)

                # Evaluate performance:
                mean_log = mean_logloss(X_val_poly, y_val, theta_opt)
                # print(f"Mean Log-Loss: {mean_log}")
                # print(
                #     f"Percentage error: {100 * classif_error(log_regr(X_val_poly, theta_opt), y_val):.2f}%"
                # )

                if mean_log < min_error:
                    min_error = mean_log
                    optimal_theta = theta_opt
                    optimal_hyper = (lr, gd_iters)
        if optimal_hyper not in param_count:
            param_count[optimal_hyper] = 0
        param_count[optimal_hyper] += 1

    # Plot data:
    (lr, iters) = max(param_count, key=param_count.get)
    print(param_count)
    print("-------------------------------------------")
    print(f"Optimal theta is: \n{optimal_theta}")
    print(
        f"With loss {mean_logloss(X_val_poly, y_val, optimal_theta)} and error {100 * classif_error(log_regr(X_val_poly, optimal_theta), y_val):.2f}%"
    )
    print(f"Optimal hyperparameters are: Learning rate = {lr}, Iterations = {iters}")
    plot_data_line(X_train, class_labels_train, optimal_theta, degree_poly)
    plot_data_line(X_val, class_labels_val, optimal_theta, degree_poly)
