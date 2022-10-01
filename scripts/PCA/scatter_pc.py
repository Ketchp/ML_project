from project.scripts.PCA.svd_pca import Vh, Y
from project.scripts.data_preparation import y
import matplotlib.pyplot as plt

Z = Y @ Vh.T

mask_under = y == 0
mask_over = y == 1

fig = plt.figure()


def create_subplot(h, w, pos, i, j):
    axis = fig.add_subplot(h, w, pos)

    handle1 = axis.scatter(Z[mask_under, i],
                           Z[mask_under, j],
                           s=1, alpha=0.6)
    handle2 = axis.scatter(Z[mask_over, i],
                           Z[mask_over, j],
                           s=1, alpha=0.3)

    axis.set_xlabel(f'PC{i+1}')
    axis.set_ylabel(f'PC{j+1}')

    return handle1, handle2


create_subplot(2, 2, 1, 0, 1)
create_subplot(2, 2, 2, 0, 2)
h1, h2 = create_subplot(2, 2, 3, 1, 2)

ax = fig.add_subplot(2, 2, 4)
ax.legend([h1, h2], ['under 50k', 'over 50k'], loc='upper left')
ax.axis('off')

plt.tight_layout()

fig.savefig('scatter_PC.png')
fig.show()
