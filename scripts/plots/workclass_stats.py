from project.scripts.data_preparation import df_nominal
import matplotlib.pyplot as plt
import numpy as np

df = df_nominal

plt.figure()

bar_w = 0.2

categories = df['workclass'].value_counts(0).index
cat_count = categories.size

under_val_c = df[df['result'] == 0]['workclass'].value_counts()
over_val_c = df[df['result'] == 1]['workclass'].value_counts()

under_bars = [under_val_c.get(cat, default=0) for cat in categories]
over_bars = [over_val_c.get(cat, default=0) for cat in categories]

x_ticks = np.arange(cat_count)

under = plt.bar(x_ticks - bar_w / 2, under_bars, width=bar_w)
over = plt.bar(x_ticks + bar_w / 2, over_bars, width=bar_w)

plt.ylim(top=20000)

cat_names = list(categories)
cat_names[1] = 'Self-emp\nnot-inc'
plt.xticks(x_ticks, cat_names, rotation=-20)

for rect in under + over:
    height = rect.get_height()
    plt.text(rect.get_x(), height, height, ha='left', va='bottom', rotation=60)

plt.legend(['under 50k', 'over 50k'])
plt.title('Work class histogram per result group')

plt.savefig('workclass_hist_per_group.png')
plt.show()
