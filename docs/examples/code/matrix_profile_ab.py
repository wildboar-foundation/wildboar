import matplotlib.pylab as plt

from wildboar.datasets import load_dataset
from wildboar.distance import matrix_profile

x, y = load_dataset("GunPoint")
mp = matrix_profile(x[0:3], x[1], window=5, exclude=0.2)

fig, ax = plt.subplots(nrows=3, sharex=True)
ax[0].plot(x[1])
for i in range(mp.shape[0]):
    ax[1].plot(x[i], label=str(i))
    ax[2].plot(mp[i], label=str(i))

ax[0].set_title("A")
ax[1].set_title("B")
ax[2].set_title("Matrix profile")
ax[0].set_xlim(0, x.shape[-1])
ax[1].set_xlim(0, x.shape[-1])
ax[2].set_xlim(0, x.shape[-1])
plt.legend()
plt.tight_layout()
plt.savefig("../fig/matrix_profile_ab.png")
