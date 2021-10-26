import matplotlib.pylab as plt
import numpy as np

from wildboar.datasets import load_dataset
from wildboar.embed import IntervalEmbedding

x, y = load_dataset("GunPoint")

fixed = IntervalEmbedding(n_interval=30, summarizer="auto", intervals="fixed")
x_t = fixed.fit_transform(x)
labels = ["%s" % start for (dim, (start, length, _)) in fixed.embedding_.features]
n_features = x_t.shape[1]

fig, ax = plt.subplots(nrows=4, figsize=(5.5, 7))
ax[0].plot(x[0])
ax[0].title.set_text("Time series")
colors = plt.cm.rainbow(np.linspace(0, 1, 3))
titles = ["Mean", "Variance", "Slope"]
for i in range(3):
    ax[i + 1].bar(labels, x_t[0, i:n_features:3], color=colors[i, :])
    plt.setp(ax[i + 1].get_xticklabels(), rotation="vertical", ha="center")
    ax[i + 1].title.set_text(titles[i])

plt.tight_layout()
# plt.xlim([0, 150])
plt.savefig("../fig/interval.png")
