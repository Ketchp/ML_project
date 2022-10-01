from project.scripts.data_preparation import df
import matplotlib.pyplot as plt
import numpy as np

df_under = df[df['result'] == 0]
df_over = df[df['result'] == 1]
N_under = df_under.shape[0]
N_over = df_over.shape[0]

noise = 0.2

fig = plt.figure()

ax = fig.add_subplot(2, 2, 1)

ax.scatter(df_under['age'],
           df_under['hours-per-week'],
           s=1, alpha=0.6)
ax.scatter(df_over['age'],
           df_over['hours-per-week'],
           s=1, alpha=0.3)
ax.set_xlabel('age')
ax.set_ylabel('hours')

ax = fig.add_subplot(2, 2, 2)
ax.scatter(df_under['age'],
           df_under['education-num'] + np.random.normal(0, noise, N_under),
           s=1, alpha=0.6)

ax.scatter(df_over['age'],
           df_over['education-num'] + np.random.normal(0, noise, N_over),
           s=1, alpha=0.3)
ax.set_xlabel('age')
ax.set_ylabel('education')

ax = fig.add_subplot(2, 2, 3)
h1 = ax.scatter(df_under['hours-per-week'] + np.random.normal(0, noise * 4, N_under),
                df_under['education-num'] + np.random.normal(0, noise, N_under),
                s=1, alpha=0.6)

h2 = ax.scatter(df_over['hours-per-week'] + np.random.normal(0, noise * 4, N_over),
                df_over['education-num'] + np.random.normal(0, noise, N_over),
                s=1, alpha=0.3)
ax.set_xlabel('hours')
ax.set_ylabel('education')

ax = fig.add_subplot(2, 2, 4)
ax.legend([h1, h2], ['under 50k', 'over 50k'], loc='upper left')
ax.axis('off')

plt.tight_layout()

fig.savefig('scatter_cont.png')
fig.show()
