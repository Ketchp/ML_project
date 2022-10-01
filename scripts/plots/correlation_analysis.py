import matplotlib.pyplot as plt
import numpy as np

from project.scripts.data_preparation import df

img = df.corr().to_numpy()

img = np.absolute(img)

plt.figure()

plt.imshow(img, cmap=plt.cm.gray)

plt.xticks([])
plt.yticks([])

plt.tight_layout()

plt.savefig('correlation_matrix.png')
plt.show()

threshold = 0.4

correlated = np.array(np.where(img > threshold)).T

attributes = df.columns
for a, b in correlated:
    if a >= b:
        continue
    print(attributes[a], ' X ', attributes[b])
