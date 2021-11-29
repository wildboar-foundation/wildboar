import matplotlib.pylab as plt

from wildboar.datasets import load_dataset
from wildboar.distance import matrix_profile

x, y = load_dataset("TwoLeadECG")
x = x[:20].reshape(-1)
print(x.shape)
mp = matrix_profile(x.reshape(-1), window=20, exclude=0.2)

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(x, color="red", lw=0.5)
ax[1].plot(mp, color="blue", lw=0.5)
ax[0].set_title("Time series")
ax[1].set_title("Matrix profile")
ax[0].set_xlim(0, x.shape[-1])
ax[1].set_xlim(0, x.shape[-1])
plt.tight_layout()
plt.savefig("../fig/matrix_profile.png")
