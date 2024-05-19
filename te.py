import numpy as np

s = np.mat([[0, 1, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

# print('s', s, type(s), s.shape)
print(s)
print(s[0])
print(s[0, :])
print(np.where(s[0, :] == 1))  # 好逆天，tmd长这样(array([0, 0, 0], dtype=int64), array([1, 4, 5], dtype=int64))
print(np.array(np.where(s[0, :] == 1)[1]))

pos = np.array(np.where(s[0, :] == 1)[1]).reshape(-1)

print(pos)

x = np.array([[[[1, 2, 3]]]])
print(x.reshape(-1))

