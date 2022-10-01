import matplotlib.pyplot as plt
from project.scripts.data_preparation import df_nominal, attributeNames
from project.scripts.PCA.svd_pca import Vh

plt.figure()

comp_idx = 0
component_name = 'First'

component = Vh.T[:, comp_idx]
plt.title(f'{component_name} principal component')

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
          'lightgreen', 'pink', 'maroon',
          'teal', 'tan', 'fuchsia']

attribute_groups = df_nominal.columns
attribute_groups = attribute_groups.drop('result')
color_map = dict(zip(attribute_groups, colors))

for key, color in color_map.items():
    x_tics = [idx for idx, attr in enumerate(attributeNames) if key in attr]
    plt.bar(x_tics, component[x_tics], color=color)

plt.xticks([])
plt.xlabel('original attribute')
plt.ylabel('attribute weight')

if comp_idx == 0:
    plt.legend(attribute_groups, ncol=2, fontsize='small')

plt.tight_layout()

plt.savefig(f'PCA_{component_name.lower()}.png')
plt.show()
