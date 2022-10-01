from project.scripts.data_preparation import df_wo_nan
import matplotlib.pyplot as plt
import numpy as np

df = df_wo_nan

f, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True)

bins = np.linspace(df['capital-gain'].min(),
                   df['capital-gain'].max(),
                   50)

ax_top.hist(df['capital-gain'], bins=bins)
ax_bottom.hist(df['capital-gain'], bins=bins)


rng = 200
ax_top.set_ylim(rng, 30000)
ax_bottom.set_ylim(top=rng)

ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.xaxis.tick_top()
ax_top.tick_params(labeltop=False)  # don't put tick labels at the top
ax_bottom.xaxis.tick_bottom()

ax_bottom.ticklabel_format(axis='x', scilimits=(0, 0))
# ax_top.ticklabel_format(axis='y', scilimits=(0, 0))

d = .01  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.0)
ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax_bottom.transAxes)  # switch to the bottom axes
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


ax_top.set_title('Histogram of capital gain')
plt.tight_layout()

plt.savefig('capital_gain_outliers.png')
plt.show()
