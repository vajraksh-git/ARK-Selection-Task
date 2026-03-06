import numpy as np

a=(np.array([1,2,3]))  

sin_values = np.sin(np.deg2rad(np.arange(0, 180, 1)))
cos_values = np.cos(np.deg2rad(np.arange(0, 180, 1)))

values = a[: , np.newaxis] * cos_values 
print(values.shape)
print(values)