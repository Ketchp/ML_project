from project.scripts.data_preparation import df
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

bar_w = 0.2

under = plt.bar(np.arange(0, 2) - bar_w / 2,
                df[df['result'] == 0]['sex'].value_counts(),
                width=bar_w)

over = plt.bar(np.arange(0, 2) + bar_w / 2,
               df[df['result'] == 1]['sex'].value_counts(),
               width=bar_w)

plt.xticks(np.arange(0, 2), ['Males', 'Females'])
plt.xlim(-0.5, 1.5)

for rect in under + over:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, height, ha='center', va='bottom')

plt.title("Sex histogram per result group")
plt.legend(['under 50k', 'over 50k'])

plt.savefig('sex_hist_per_group.png')
plt.show()
