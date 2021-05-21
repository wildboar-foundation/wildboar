from wildboar.datasets import load_dataset
from wildboar.tree import ShapeletTreeRegressor
from wildboar.ensemble import ShapeletForestRegressor
from sklearn.metrics import mean_squared_error

x_train, x_test, y_train, y_test = load_dataset(
    "FloodModeling3",
    repository="wildboar/tsereg",
    merge_train_test=False,
    preprocess=None,
)
f = ShapeletForestRegressor(
    metric="euclidean",
    n_estimators=100,
    n_jobs=-1,
    min_shapelet_size=0.9,
    max_shapelet_size=1.0,
    random_state=123,
)
f.fit(x_train, y_train)

print("%.5f" % mean_squared_error(y_test, f.predict(x_test), squared=False))
