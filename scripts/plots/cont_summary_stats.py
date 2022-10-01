from project.scripts.data_preparation import df
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

header = ['Mean', 'Median', 'Minimum', 'Maximum', 'Standard deviation', 'Mode']

attributes = ['age', 'education-num', 'capital-gain', 'hours-per-week']
attribute_repr = ['Age', 'Education number', 'Capital gain', 'Working hours']

stat_func = [np.mean, np.median, np.min, np.max, np.std, lambda d: stats.mode(d, keepdims=True)[0][0]]


def format_func(n):
    if type(n) == np.float64:
        return f'{n:.2f}'
    return str(n)


raw_table = [[format_func(func(df[attr])) for func in stat_func] for attr in attributes]

plt.figure(linewidth=1,
           tight_layout={'pad': 1},
           figsize=(10, 2))

header_col = plt.cm.BuPu(np.full(len(header), 0.1))
row_col = plt.cm.BuPu(np.full(len(attributes), 0.1))

tbl = plt.table(raw_table,
                rowLabels=attribute_repr,
                colLabels=header,
                rowLoc='right',
                loc='center',
                rowColours=row_col,
                colColours=header_col)

tbl.scale(1, 1.5)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.box(on=None)

plt.tight_layout()

plt.suptitle("Summary statistics of continuous data")
plt.draw()


plt.savefig('cont_summary_stats.png', dpi=250)
plt.show()
