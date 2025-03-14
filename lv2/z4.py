import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50,50))
white = np.ones((50,50))

row1 = np.hstack((black, white))
row2 = np.hstack((white, black))

board = np.vstack((row1, row2))

plt.imshow(board, cmap="gray")
plt.show()