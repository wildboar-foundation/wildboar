import numpy as np
import time

from sklearn.ensemble import BaggingClassifier
from wildboar.tree import ShapeletTreeClassifier
from wildboar._utils import print_tree

x = [
    # Example 1
    [
        [0, 0, 1, 10, 1],  # dimension 1
        [0, 0, 1, 10, 1]
    ],  # dimension 2 

    # Example 2
    [
        [0, 1, 9, 1, 0],  # dimension 1
        [1, 9, 1, 0, 0]
    ],  # dimension 2 

    # etc...
    [[0, 1, 9, 1, 0], [0, 1, 2, 3, 4]],
    [[1, 2, 3, 0, 0], [0, 0, 0, 1, 2]],
    [[0, 0, -1, 0, 1], [1, 2, 3, 0, 1]],
]

# `x` is an array of shape `[5, 2, 5]`, i.e., 5 examples with 2
# dimensions consisting of 5 timesteps
x = np.array(x, dtype=np.float64)

n_samples, n_dimensions, n_timesteps = x.shape

# `y` is the output target
y = np.array([0, 0, 1, 1, 0])

random_state = np.random.RandomState(123)
order = np.arange(n_samples)
random_state.shuffle(order)

x = x[order, :, :]
y = y[order]

tree = ShapeletTreeClassifier(
    random_state=10,
    # due to `BaggingClassifier` requiring a 2d-array the
    # trees need to reshape the input data
    force_dim=n_dimensions,
    metric="scaled_euclidean",
)

bag = BaggingClassifier(
    base_estimator=tree,
    bootstrap=True,
    n_jobs=16,
    n_estimators=100,
    random_state=100,
)

c = time.time()

# to use the `BaggingClassifier` reshape to a 2d-array. The trees will
# reshape the data if `force_dim` is properly set
x = x.reshape(n_samples, n_dimensions * n_timesteps)
bag.fit(x, y)

predict = 1
print("Predicting the class of example:")
print(x[predict, :].reshape(-1, n_dimensions, n_timesteps))
print("The true class is:", y[predict], "and the predicted class is:",
      bag.predict(x[predict, :].reshape(1, -1))[0])
