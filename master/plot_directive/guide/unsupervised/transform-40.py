import math

def binsearch_depth(i):

   low = 0
   high = math.ceil(math.log2(i + 2))

   while low < high:
      mid = (low + high) // 2
      if 2 ** (mid + 1) - 2 - mid > i:
         high = mid
      else:
         low = mid + 1
   return low

fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

step = 1.0 / f.depth
n_first = 2**f.depth - 1
for i, (_, (start, length, _)) in enumerate(f.embedding_.attributes[:n_first]):
   end = start + length
   depth = math.floor(math.log2(i + 1))
   ax[0].axvspan(
      start,
      end,
      ymin=1 - (step * depth) - 0.02,
      ymax=1 - (step * (depth + 1)),
      facecolor="gray",
      edgecolor="black",
      alpha=0.1,
   )

for i, (_, (start, length, _)) in enumerate(f.embedding_.attributes[n_first:]):
   end = start + length
   depth = binsearch_depth(i)
   ax[1].axvspan(
      start,
      end,
      ymin=1 - (step * depth) - 0.02,
      ymax=1 - (step * (depth + 1)),
      facecolor="gray",
      edgecolor="black",
      alpha=0.1,
   )

ax[0].plot(X_train[0])
ax[1].plot(X_train[0])