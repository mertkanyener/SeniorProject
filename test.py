import numpy as np


A = [[1,2,3], [3,4,5], [5,6,7], [7,8,9]]
A = np.asanyarray(A).reshape(4,3)
print(A.shape)