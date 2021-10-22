import matplotlib.pylab as plt
import pandas as pd

from wildboar.datasets import load_dataset
from wildboar.embed import IntervalEmbedding

x, y = load_dataset("GunPoint")

fixed = IntervalEmbedding(n_interval=30, summarizer="auto", intervals="fixed")
x_t = fixed.fit_transform(x)
labels = ["%s" % start for (dim, (start, length, _)) in fixed.embedding_.features]
n_features = x_t.shape[1]
df = pd.DataFrame(
    {
        "mean": x_t[0, 0:n_features:3],
        "variance": x_t[0, 1:n_features:3],
        "slope": x_t[0, 2:n_features:3],
    },
    index=labels,
)
fig, ax = plt.subplots(nrows=2)
ax[0].plot(x[0])
df.plot(kind="bar", ax=ax[1])
plt.savefig("../fig/interval.png")
