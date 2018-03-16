import pypf._sliding_distance
import numpy as np

s = np.random.randn(100)#np.array([10, 20, 30], dtype=np.float64)
t = np.random.randn(1000) #np.array([0, 0, 0, 100, 100, 100, 10, 20, 30], dtype=np.float64)
s = (s-np.mean(s)) / np.std(s)
# print(s)
dist = pypf._sliding_distance.sliding_distance(s, t)
#print(dist)

x = np.random.randn(10, 10)
y = np.random.randint(0, 2, size=10)
from pypf.tree import partition
#print(y)

d = np.array([2,3,4,5,7,9])
y = np.array([1,1,1,0,0,1])
partition(d, y, 2)
