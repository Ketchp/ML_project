from project.scripts.data_preparation import df, df_wo_nan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


f, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True)

_, _, patches_top = ax_top.hist(df_wo_nan['native-country'])
_, _, patches_bottom = ax_bottom.hist(df_wo_nan['native-country'])

patches_top[0].set_facecolor('r')
patches_bottom[0].set_facecolor('r')

rng = 2000
ax_top.set_ylim(26000, 26000 + rng)
ax_bottom.set_ylim(0, rng)

ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)

ax_top.set_xticks([])
ax_bottom.set_xticks([])

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax_bottom.transAxes)  # switch to the bottom axes
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax_top.legend([Line2D([0], [0], color='r', lw=4),
               Line2D([0], [0], color='b', lw=4)],
              ['US', 'non US'])

ax_top.set_title('Histogram of all native countries')

plt.savefig('native_countries_hist.png')
plt.show()

bar_w = 0.2

under = plt.bar(np.arange(0, 2) - bar_w / 2,
                df[df['result'] == 0]['from-US'].value_counts(),
                width=bar_w)

over = plt.bar(np.arange(0, 2) + bar_w / 2,
               df[df['result'] == 1]['from-US'].value_counts(),
               width=bar_w)

plt.xticks(np.arange(0, 2), ['from US', 'other countries'])
plt.xlim(-0.5, 1.5)

for rect in under + over:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, height, ha='center', va='bottom')

plt.title("Native country histogram per result group")
plt.legend(['under 50k', 'over 50k'])

plt.savefig('native_countries_aggr_hist.png')
plt.show()
