import numpy as np
import matplotlib.pylab as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from wildboar.datasets import load_two_lead_ecg
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain.counterfactual import counterfactuals


def counterfactuals_plot(estimator, method="infer"):
    x, y = load_two_lead_ecg()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=123
    )
    print("fitting the estimator %r" % estimator)
    estimator.fit(x_train, y_train)

    x_test = x_test[y_test == 2.0]

    print("computing %d counterfactuals" % x_test.shape[0])
    x_counterfactuals, success, score = counterfactuals(
        estimator, x_test, 1.0, scoring="euclidean", method=method, random_state=123
    )

    x_test = x_test[success]
    x_counterfactuals = x_counterfactuals[success]
    i = np.argsort(score[success])[:2]
    x_counterfactuals = x_counterfactuals[i, :]
    x_test = x_test[i, :]

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6, 2.5))
    ax[0].plot(x_test[0, :], c="red")
    ax[0].plot(x_counterfactuals[0, :], c="blue")
    ax[1].plot(x_test[1, :], c="red")
    ax[1].plot(x_counterfactuals[1, :], c="blue")
    ax[1].legend(["y=abnormal", "y*=normal"])
    return fig


def counterfactuals_nn():
    return counterfactuals_plot(KNeighborsClassifier(n_neighbors=5, metric="euclidean"))


def counterfactuals_sf():
    return counterfactuals_plot(
        ShapeletForestClassifier(random_state=123, n_jobs=-1, metric="euclidean")
    )


PLOT_DICT = {
    "counterfactuals_nn": counterfactuals_nn,
    "counterfactuals_sf": counterfactuals_sf,
}
