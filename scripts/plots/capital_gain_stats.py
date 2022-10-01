from project.scripts.data_preparation import df
import matplotlib.pyplot as plt
import numpy as np

bin_count = 100
bin_edges = np.linspace(df['capital-gain'].min(),
                        df['capital-gain'].max() + 1,
                        bin_count)

plt.figure()

plt.hist(df['capital-gain'], bins=bin_edges)

plt.title('Capital gain/loss histogram')
plt.xlabel('capital gain/loss')
plt.ylim(top=1200)

plt.savefig('capital_gain_loss_hist.png')
plt.show()

# per group plot
df.groupby('result')['capital-gain'].hist(bins=bin_edges, alpha=0.6, grid=False)

plt.title('Capital gain/loss histogram per result group')
plt.xlabel('capital gain/loss')
plt.legend(['under 50k', 'over 50k'])
plt.xlim(bin_edges[0] - 5,
         bin_edges[-1] + 5)
plt.ylim(top=1200)
plt.tight_layout()

plt.savefig('capital_gain_loss_hist_per_group.png')
plt.show()
