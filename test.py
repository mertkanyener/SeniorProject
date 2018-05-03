import numpy as np


A = [3,4,5]
b = [8,9]
d = [10, 1,3]

c = np.concatenate((A, b, d), axis=0)
print(c)