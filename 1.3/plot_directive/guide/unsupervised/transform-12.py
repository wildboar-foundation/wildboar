import matplotlib.pylab as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_transform)

for label in  [1, 2]:
   plt.scatter(
      X_test_pca[y_test == label, 0],
      X_test_pca[y_test == label, 1],
      label=f"Label {label}",
   )

plt.xlabel("Component 0")
plt.ylabel("Component 1")
plt.legend()