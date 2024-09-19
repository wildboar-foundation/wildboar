fig, ax = plt.subplots()
for _, (start, length, _) in f.embedding_.attributes:
   end = start + length
   ax.axvspan(
      start,
      end,
      ymin=0.02,
      ymax=0.98,
      facecolor="gray",
      edgecolor="black",
      alpha=0.1,
   )

ax.plot(X_train[0])