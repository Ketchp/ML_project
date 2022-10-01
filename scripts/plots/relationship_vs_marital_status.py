import matplotlib.pyplot as plt
import numpy as np
from project.scripts.data_preparation import df_nominal


rel_catg = df_nominal['relationship'].unique()
mar_catg = df_nominal['marital-status'].unique()

rel_cnt = len(rel_catg)
mar_cnt = len(mar_catg)

rel_map = dict(zip(rel_catg, range(rel_cnt)))
mar_map = dict(zip(mar_catg, range(mar_cnt)))


df_nominal['relationship'].replace(rel_map, inplace=True)
df_nominal['marital-status'].replace(mar_map, inplace=True)

N = df_nominal.shape[0]

plt.figure()

# plt.hist2d(df_nominal['relationship'],
#            df_nominal['marital-status'],
#            bins=[np.array(range(rel_cnt+1))-0.5,
#                  np.array(range(mar_cnt+1))-0.5])
# fig_name = 'rel-vs-mar_st-hist2d.png'

# V this V or ^ this ^

df_nominal['relationship'] += np.random.normal(0, 0.1, N)
df_nominal['marital-status'] += np.random.normal(0, 0.1, N)
plt.scatter(df_nominal['relationship'],
            df_nominal['marital-status'],
            s=1)
fig_name = 'rel-vs-mar_st-scatter.png'

plt.xticks(range(rel_cnt),
           [name.replace('-', ' ') for name in rel_catg])
plt.yticks(range(mar_cnt),
           [name.replace('-', '\n') for name in mar_catg])

plt.xlabel('relationship')
plt.ylabel('marital status')

plt.tight_layout()

plt.savefig(fname=fig_name)
plt.show()
