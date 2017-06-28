import numpy as np
from operator import itemgetter

example = np.array([
    [ [1, 1], [0, 2], [1, 1], [2, 0] ]
])
print(example)
print()

changed = np.reshape(example, (2, -1, 4))
x1 = example[:, :, 0]
x2 = example[:, :, 1]
print(x1)
print(x2)

result = np.concatenate((x1, x2))
print()
print(result)
print()
result = np.reshape(result, (-1, 2, 4), order="F")
print(result)