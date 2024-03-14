plt.figure(figsize=(12, 4))
plt.plot(X[0])
cmap = plt.get_cmap(lut=len(unique))
for i, (start, end) in enumerate(segments):
   plt.axvspan(start, end, 0, 1, alpha=0.1, color=cmap(i))