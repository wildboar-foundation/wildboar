from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from wildboar.datasets import load_dataset
from wildboar.ensemble import RockestRegressor, ShapeletForestRegressor
from wildboar.linear_model import RocketRegressor


def test(dataset, f, random_state=None):
    x, y = load_dataset(
        dataset,
        repository="wildboar/tsereg",
        merge_train_test=True,
        preprocess=None,
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)
    f.fit(x_train, y_train)
    print("%.6f" % mean_squared_error(y_test, f.predict(x_test), squared=False))


test(
    "FloodModeling1",
    ShapeletForestRegressor(
        metric="euclidean",
        n_estimators=100,
        n_jobs=-1,
        min_shapelet_size=0.9,
        max_shapelet_size=1.0,
        random_state=123,
    ),
    random_state=123,
)

test(
    "FloodModeling1",
    RocketRegressor(
        n_jobs=-1,
        normalize=True,
        random_state=123,
    ),
    random_state=123,
)

test(
    "FloodModeling1",
    RockestRegressor(
        n_kernels=100,
        n_jobs=-1,
        random_state=123,
    ),
    random_state=123,
)
