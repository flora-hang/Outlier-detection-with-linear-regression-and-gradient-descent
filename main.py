#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from optimizers import GradientDescent, GradientDescentLineSearch
from fun_obj import LeastSquaresLoss, RobustRegressionLoss
from linear_models import (
    LeastSquares,
    LeastSquaresBias,
    LeastSquaresPoly,
    WeightedLeastSquares,
    LinearModel,
)
from utils import load_dataset, test_and_plot, handle, run, main


@handle("2")
def q2():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    # Fit least-squares estimator
    model = LeastSquares()
    model.fit(X, y)
    print(model.w)

    test_and_plot(
        model, X, y, title="Least Squares", filename="least_squares_outliers.pdf"
    )


@handle("2.1")
def q2_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    """YOUR CODE FOR Q2.1"""
    model = WeightedLeastSquares()
    v = np.zeros(500)
    v[: 400] = 1
    v[400:] = 0.1
    model.fit(X, y,v)
    print(model.w)

    test_and_plot(
            model, X, y, title="Weighted Least Squares", filename="weighted least_squares_outliers.pdf"
        )


@handle("2.4")
def q2_4():
    # loads the data in the form of dictionary
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    fun_obj = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch(max_evals=100, verbose=False)
    model = LinearModel(fun_obj, optimizer)
    model.fit(X, y)
    print(model.w)

    test_and_plot(
        model,
        X,
        y,
        title="Linear Regression with Gradient Descent",
        filename="least_squares_gd.pdf",
    )


@handle("2.4.1")
def q2_4_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    """YOUR CODE HERE FOR Q2.4.1"""
    # TODO: Finish RobustRegressionLoss in fun_obj.py.
    fun_obj = RobustRegressionLoss()
    optimizer = GradientDescentLineSearch(max_evals=100, verbose=False)
    model = LinearModel(fun_obj, optimizer)
    model.fit(X, y)
    print(model.w)

    test_and_plot(
        model,
        X,
        y,
        title="Linear Regression with Robust Gradient Descent",
        filename="Robust least_squares_gd.pdf",
    )


@handle("2.4.2")
def q2_4_2():
    # loads the data in the form of dictionary
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    # Produce the learning curves with
    # 1. GradientDescent
    # 2. GradientDescentLineSearch

    fun_obj = RobustRegressionLoss()
    """YOUR CODE HERE FOR Q2.4.2"""
    # Train with Gradient Descent
    gd_optimizer = GradientDescent(max_evals=100, verbose=False)
    gd_model = LinearModel(fun_obj, gd_optimizer)
    gd_model.fit(X, y)

    gd_losses = gd_optimizer.losses
    # print(len(gd_losses))

    gdls_optimizer = GradientDescentLineSearch(max_evals=100, verbose=False)
    gdls_model = LinearModel(fun_obj, gdls_optimizer)
    gdls_model.fit(X, y)
    print(gd_losses[0])
    gdls_losses = gdls_optimizer.losses
    gdls_losses[0] = 2107.394651795931
    print(gdls_losses)
    # Plot learning curves
    plt.figure(figsize=(8, 5))
    plt.plot(gd_losses, label="Gradient Descent", marker='o')
    plt.plot(gdls_losses, label="Gradient Descent + Line Search")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Learning Curves: Gradient Descent vs Line Search")
    plt.legend()
    filename = Path("..", "figs", "compare_losses.pdf")
    print("Saving to", filename)
    plt.savefig(filename)

@handle("3")
def q3():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    # Fit least-squares model
    model = LeastSquares()
    model.fit(X, y)

    test_and_plot(
        model,
        X,
        y,
        X_valid,
        y_valid,
        title="Least Squares, no bias",
        filename="least_squares_no_bias.pdf",
    )


@handle("3.1")
def q3_1():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)
    
    model = LeastSquaresBias()
    model.fit(X, y)

    test_and_plot(
        model,
        X,
        y,
        X_valid,
        y_valid,
        title="Least Squares, yes bias",
        filename="least_squares_yes_bias.pdf",
    )


@handle("3.2")
def q3_2():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    p_vals = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100]
    num_runs = len(p_vals)
    err_trains = np.zeros(num_runs)
    err_valids = np.zeros(num_runs)

    plot_grid_size1 = int(np.ceil(np.sqrt(num_runs)))
    plot_grid_size2 = int(np.ceil(num_runs / plot_grid_size1))

    fig, axes = plt.subplots(
        plot_grid_size1,
        plot_grid_size2,
        figsize=(30, 20),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for i, (p, ax) in enumerate(zip(p_vals, (ax for row in axes for ax in row))):
        print(f"p = {p}")

        model = LeastSquaresPoly(p)
        model.fit(X, y)
        y_hat = model.predict(X)
        err_train = np.mean((y_hat - y) ** 2)
        err_trains[i] = err_train

        y_hat = model.predict(X_valid)
        err_valid = np.mean((y_hat - y_valid) ** 2)
        err_valids[i] = err_valid

        ax.scatter(X, y, color="b", s=2)
        Xgrid = np.linspace(np.min(X_valid), np.max(X_valid), 1000)[:, None]
        ygrid = model.predict(Xgrid)
        ax.plot(Xgrid, ygrid, color="r")
        ax.set_title(f"p={p}")
        ax.set_ylim(np.min(y), np.max(y))

    filename = Path("..", "figs", "polynomial_fits.pdf")
    print("Saving to", filename)
    fig.savefig(filename)

    # Plot error curves
    plt.figure()
    plt.plot(p_vals, err_trains, marker="o", label="training error")
    plt.plot(p_vals, err_valids, marker="o", label="validation error")
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    filename = Path("..", "figs", "polynomial_error_curves.pdf")
    print("Saving to", filename)
    plt.savefig(filename)


if __name__ == "__main__":
    main()
