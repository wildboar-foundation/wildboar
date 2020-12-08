import numpy as np

from wildboar.ensemble import ShapeletForestClassifier

# Example 2: multivariate shapelet forest

# fmt:off
x = [
    # Example 1
    [
        [0, 0, 1, 10, 1],  # dimension 1
        [0, 0, 1, 10, 1],  # dimension 2
    ],

    # Example 2
    [
        [0, 1, 9, 1, 0],  # dimension 1
        [1, 9, 1, 0, 0],  # dimension 2
    ],

    # etc...
    [[0, 1, 9, 1, 0], [0, 1, 2, 3, 4]],
    [[1, 2, 3, 0, 0], [0, 0, 0, 1, 2]],
    [[0, 0, -1, 0, 1], [1, 2, 3, 0, 1]],
]
# fmt:on

# `x` is an array of shape `[5, 2, 5]`, i.e., 5 examples with 2
# dimensions each consisting of 5 time steps
x = np.array(x, dtype=np.float64)
n_samples, n_dimensions, n_timesteps = x.shape

# `y` is the output target
y = np.array([0, 0, 1, 1, 0])

random_state = np.random.RandomState(123)
order = np.arange(n_samples)
random_state.shuffle(order)

x = x[order, :, :]
y = y[order]

f = ShapeletForestClassifier(random_state=random_state, n_shapelets=1)
f.fit(x, y)

predict = 1
print("Predicting the class of example:")
print(x[predict, :].reshape(-1, n_dimensions, n_timesteps))
print(
    "The true class is:",
    y[predict],
    "and the predicted class is:",
    f.predict(x[predict, :].reshape(1, n_dimensions, -1))[0],
)
