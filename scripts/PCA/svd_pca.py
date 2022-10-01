from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from project.scripts.data_preparation import X, N

# data normalisation
Y = X - np.ones((N, 1)) * X.mean(axis=0)
Y *= 1 / np.std(Y, 0)

U, S, Vh = svd(Y, full_matrices=False)

rho = S*S / (S*S).sum()

cumulative_rho = np.cumsum(rho)

threshold = 0.90
PC_required = np.argmax(cumulative_rho > threshold)

if __name__ == "__main__":
    plt.figure()
    plt.plot(range(1, len(rho)+1), rho, 'x-')
    plt.plot(range(1, len(rho)+1), cumulative_rho, 'o-', markersize=4)
    plt.plot([1, len(rho)], [threshold] * 2, 'b--')
    plt.plot([PC_required] * 2, [0, 1], 'k--')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual', 'Cumulative', 'Threshold', 'Required PC'])
    plt.grid()

    plt.tight_layout()

    plt.savefig('variation_explained.png')
    plt.show()
