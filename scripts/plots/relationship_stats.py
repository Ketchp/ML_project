from project.scripts.data_preparation import df_nominal
import matplotlib.pyplot as plt
import numpy as np

df = df_nominal

bar_w = 0.2

categories = df['relationship'].value_counts().index
cat_count = categories.size

x_ticks = np.arange(cat_count)

under = plt.bar(x_ticks - bar_w / 2,
                df[df['result'] == 0]['relationship'].value_counts(),
                width=bar_w)

over = plt.bar(x_ticks + bar_w / 2,
               df[df['result'] == 1]['relationship'].value_counts(),
               width=bar_w)

cat_names = [name.replace('-', ' ') for name in categories]
plt.xticks(x_ticks, cat_names)

for rect in under:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, height, ha='center', va='bottom')

for rect in over:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 8.0, height, height, ha='left', va='bottom')

plt.title("Relationship histogram per result group")
plt.legend(['under 50k', 'over 50k'])

plt.savefig('relationship_per_group.png')
plt.show()
