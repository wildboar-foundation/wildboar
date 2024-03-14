fig, ax = plt.subplots(figsize=(8, 4), sharex=True)
plot_time_domain(X, ax=ax)
cmap = plt.get_cmap(lut=1)
for c, ind in enumerate(motif_indices):
   first = ind[0]
   for i in ind:
      ax.plot(np.arange(i, i + 200), X[i : i + 200], color=cmap(c))

   ax.annotate(
      f"{c}",
      xy=(first, X[first]),
      xytext=(first, X[first] - 2000),
      arrowprops=dict(arrowstyle="->"),
   )