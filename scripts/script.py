from scipy.linalg import svd
import matplotlib.pyplot as plt
from data_preparation import *

print(X.mean(axis=0))

# data normalisation
Y = X - np.ones((N, 1)) * X.mean(axis=0)


print(np.std(Y, 0))
Y *= 1 / np.std(Y, 0)


U, S, Vh = svd(Y, full_matrices=False)

rho = S*S / (S*S).sum()

cumulative_rho = np.cumsum(rho)

threshold = 0.9
PC_required = np.argmax(cumulative_rho > threshold)
print(f'Required principal components: {PC_required}')

plt.figure()
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), cumulative_rho, 'o-')
plt.plot([1, len(rho)], [threshold] * 2, 'b--')
plt.plot([PC_required] * 2, [0, 1], 'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold', 'Required PC'])
plt.grid()
plt.show()

component = Vh.T[:, 1]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
          'lightgreen', 'pink', 'maroon',
          'teal', 'tan', 'fuchsia']
attribute_groups = names.copy()
attribute_groups.drop(columns=['education', 'result'], inplace=True)
color_map = dict(zip(attribute_groups, colors))

plt.figure()
for key, color in color_map.items():
    x_tics = [idx for idx, attr in enumerate(attributeNames) if key in attr]
    plt.bar(x_tics, component[x_tics], color=color)

plt.legend(attribute_groups, ncol=2, fontsize='small')
plt.show()
