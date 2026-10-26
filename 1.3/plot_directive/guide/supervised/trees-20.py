import numpy as np
import matplotlib.pylab as plt

fig = plt.figure(figsize=(8, 4))
arrays = [(i, attr[1][1]) for i, attr in enumerate(clf.tree_.attribute) if attr is not None]
spec = fig.add_gridspec(len(arrays), 5)

left = fig.add_subplot(spec[:, :4])
plot_tree(clf, ax=left, fontsize=None)

cmap = plt.get_cmap(lut=len(arrays))
for i, (index, arr) in enumerate(arrays):
   ax = fig.add_subplot(spec[i, 4])
   ax.plot(arr, label=f"S[{index}]", color=cmap(i))
   ax.set_xlim([len(arr)-X_test.shape[1], X_test.shape[1]])
   ax.set_ylim([np.min(X_train), np.max(X_train)])
   ax.set_axis_off()

fig.legend()