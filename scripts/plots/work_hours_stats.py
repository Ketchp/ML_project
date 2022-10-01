from project.scripts.data_preparation import df
import matplotlib.pyplot as plt
import numpy as np


bin_edges = np.arange(df['hours-per-week'].min(),
                      df['hours-per-week'].max() + 1)

plt.figure()

plt.hist(df['hours-per-week'], bins=bin_edges)

plt.title('Working hours histogram')
plt.xlabel('hours per week')
plt.xlim(bin_edges[0] - 1,
         bin_edges[-1] + 1)
plt.tight_layout()

plt.savefig('work_hours_hist.png')
plt.show()

# per group plot
df.groupby('result')['hours-per-week'].plot(kind='kde')

plt.title('Working hours density estimation per result group')
plt.xlabel('hours per week')
plt.legend(['under 50k', 'over 50k'])
plt.xlim(bin_edges[0] - 1,
         bin_edges[-1] + 1)

plt.yticks([])
plt.ylim(bottom=0)
plt.tight_layout()

plt.savefig('work_hours_kde.png')
plt.show()
