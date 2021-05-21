from wildboar.datasets import load_dataset
from wildboar.linear_model import RocketRegressor, RandomShapeletRegressor
from wildboar.ensemble import ShapeletForestRegressor
from sklearn.metrics import mean_squared_error


def test(dataset, f):
    x_train, x_test, y_train, y_test = load_dataset(
        dataset,
        repository="wildboar/tsereg",
        merge_train_test=False,
        preprocess=None,
    )
    f.fit(x_train, y_train)
    print("%.6f" % mean_squared_error(y_test, f.predict(x_test), squared=False))


test(
    "FloodModeling1",
    ShapeletForestRegressor(
        metric="euclidean",
        n_estimators=1,
        n_jobs=-1,
        min_shapelet_size=0.9,
        max_shapelet_size=1.0,
        random_state=123,
    ),
)
test(
    "NewsTitleSentiment",
    RocketRegressor(n_jobs=-1, n_kernels=100, normalize=True, random_state=123),
)
