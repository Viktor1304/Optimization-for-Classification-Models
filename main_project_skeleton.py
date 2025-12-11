from matplotlib.lines import Line2D
from create_data import create_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams


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


def grad_descent(X_train, y_train, learning_rate: float, grad, iters_total: int):
    theta = np.zeros((X_train.shape[1], 1))

    N = X_train.shape[0]

    loss_arr = []
    iter_arr = []

    for iter in range(iters_total):
        loss_arr.append(mean_logloss(X_train, y_train, theta))
        iter_arr.append(iter)
        theta -= learning_rate * grad(X_train, y_train, theta) / N

    return theta, loss_arr, iter_arr


def mean_logloss(X, y_real, theta):
    y_pred = sigmoid(X, theta).reshape(-1)
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

    return (1.0 - np.mean(y_hat == y_pred)) * 100.0


def create_features_for_poly(X, degree: int):
    x1 = X[:, 0]
    x2 = X[:, 1]

    features = []

    for d in range(degree, -1, -1):
        for i in range(d, -1, -1):
            j = d - i
            features.append((x1**i) * (x2**j))

    return np.vstack(features).T


def plot_data(x, class_labels) -> None:
    """
    Plots the data returned from the create_data() function.
    x: Matrix of dimensions number_of_samples x number_of_features.
       This should NOT include the concatenated 1 for the bias.
    class_labels: Vector of dimensions number_of_samples.
                  Expects values class_labels={1,2} . Not the y={0,1}
    """
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
    ax.grid(True)

    plt.show()


def plot_data_line(x, class_labels, theta, degree: int, n_iters=0) -> None:
    mpl.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        x[class_labels == 1, 0],
        x[class_labels == 1, 1],
        s=50,
        c="#d62728",  # red
        edgecolors="black",
        linewidth=0.8,
        label="Class 1",
    )

    ax.scatter(
        x[class_labels == 2, 0],
        x[class_labels == 2, 1],
        s=50,
        c="#2ca02c",  # green
        edgecolors="black",
        linewidth=0.8,
        label="Class 2",
    )

    x1_vals = np.linspace(-2, 3, 400)
    x2_vals = np.linspace(-2, 3, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    grid = np.column_stack([X1.ravel(), X2.ravel()])
    Phi = create_features_for_poly(grid, degree)
    Z = (Phi @ theta).reshape(X1.shape)

    ax.contour(
        X1,
        X2,
        Z,
        levels=[0],
        colors="#00008B",
        linewidths=2.2,
    )

    boundary_proxy = Line2D(
        [], [], color="#00008B", linewidth=2.2, label="Decision boundary"
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim((-2.0, 3.0))
    ax.set_ylim((-2.0, 3.0))

    handles, labels = ax.get_legend_handles_labels()
    handles.append(boundary_proxy)
    labels.append("Decision boundary")
    ax.legend(handles, labels, frameon=True, framealpha=0.95, edgecolor="0.7")

    plt.tight_layout()
    plt.title(f"GD Iterations = {n_iters}")
    plt.show()
    # plt.savefig(f"{n_iters}_poly3.pdf")


rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.figsize": (6, 4),
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def plot_loss_curve(
    loss_history, loss_iterations, filename: str = "loss_curve.pdf"
) -> None:
    """
    loss_history : list or array of loss values at each GD iteration
    """

    plt.figure()
    plt.plot(loss_iterations, loss_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Log-Loss")
    plt.title("Gradient Descent Convergence")
    plt.tight_layout()

    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Hyper-parameters:
    lr = 0.5
    gd_iters: int = 1000
    degree_poly = 3
    global_iterations = 1
    n_samples_train = 400

    val_error_arr = []
    train_error_arr = []
    sigma_val = 0.0
    sigma_train = 0.0
    for iteration in range(global_iterations):
        # Create training data
        [X_train, class_labels_train] = create_data(n_samples_train)

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

        # Optimize - Logistic Regression - Gradient Descent
        theta_opt, loss_history, iterations = grad_descent(
            X_train_poly, y_train, lr, log_grad, gd_iters
        )

        # Evaluate performance:
        mean_log = mean_logloss(X_val_poly, y_val, theta_opt)
        train_error = classif_error(log_regr(X_train_poly, theta_opt), y_train)
        train_error_arr.append(train_error)
        val_error = classif_error(log_regr(X_val_poly, theta_opt), y_val)
        val_error_arr.append(val_error)
        # val_error += mean_log

        # Plot data:
        # print("-------------------------------------------")
        # print(f"Optimal theta is: \n{theta_opt}")
        # print(
        #     f"With loss {mean_logloss(X_val_poly, y_val, theta_opt)} and error {100 * classif_error(log_regr(X_val_poly, theta_opt), y_val):.2f}%"
        # )
        # print(
        #     f"Optimal hyperparameters are: Learning rate = {lr}, Iterations = {gd_iters}"
        # )
        plot_data_line(
            X_train, class_labels_train, theta_opt, degree_poly, n_iters=gd_iters
        )
        # if iteration == global_iterations - 1:
        #     plot_data_line(
        #         X_val,
        #         class_labels_val,
        #         theta_opt,
        #         degree_poly,
        #         n_iters=n_samples_train,
        #     )
        # plot_loss_curve(loss_history, iterations)

    train_error = sum(train_error_arr) / global_iterations
    val_error = sum(val_error_arr) / global_iterations

    sigma_train = sum([(x - train_error) ** 2 for x in train_error_arr])
    sigma_train = (sigma_train / global_iterations) ** 0.5

    sigma_val = sum([(x - val_error) ** 2 for x in val_error_arr])
    sigma_val = (sigma_val**0.5) / global_iterations

    print(f"Training steps: {n_samples_train}")
    print(f"Average prediction error training: {train_error:.2f}%")
    print(f"Standard deviation validation: {sigma_val:.2f}")
    print(f"Average prediction error validation: {val_error:.2f}%")
    print(f"Standard deviation train: {sigma_train:.2f}")
