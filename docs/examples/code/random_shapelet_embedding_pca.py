import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from wildboar.datasets import load_dataset
from wildboar.embed import RandomShapeletEmbedding

random_state = 1234
x, y = load_dataset("CBF")

pca = make_pipeline(
    RandomShapeletEmbedding(
        metric="scaled_euclidean",
        random_state=random_state,
    ),
    PCA(n_components=2, random_state=random_state),
)
p = pca.fit_transform(x)
var = pca.steps[1][1].explained_variance_ratio_

labels, index = np.unique(y, return_inverse=True)
colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
plt.scatter(p[:, 0], p[:, 1], color=colors[index, :])
plt.xlabel("Component 1 (%.2f variance explained)" % var[0])
plt.ylabel("Component 2 (%.2f variance explained)" % var[1])
plt.savefig("../fig/rse_pca.png")
