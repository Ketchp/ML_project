from project.scripts.data_preparation import df_nominal
import matplotlib.pyplot as plt
import numpy as np

df = df_nominal

header = ['Under 50k', 'Over 50k', 'Total']

attributes = ['Craft-repair', 'Prof-specialty', 'Exec-managerial', 'Adm-clerical',
              'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving',
              'Handlers-cleaners', 'Farming-fishing', 'Tech-support',
              'Protective-serv', 'Priv-house-serv', 'Armed-Forces']

attribute_repr = ['Craft repair', 'Professional specialty', 'Executive/Managerial', 'Administrative/Clerical',
                  'Sales', 'Other services', 'Machine operator/Inspector', 'Transportation',
                  'Handlers/Cleaners', 'Farming/Fishing', 'Tech support',
                  'Protective services', 'Private house services', 'Armed forces']


raw_table = np.array([df[df['result'] == res]['occupation'].value_counts().get(attributes).to_numpy() for res in (0,
                                                                                                                  1)])

raw_table = np.append(raw_table, [raw_table[0] + raw_table[1]], axis=0).T

plt.figure(linewidth=2,
           tight_layout={'pad': 1},
           figsize=(10, 5))

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

plt.suptitle("Occupation statistics")
plt.draw()

plt.savefig('occupation_table.png', dpi=250)
plt.show()


bar_w = 0.2

categories = df['occupation'].value_counts(0).index
cat_count = categories.size

under_val_c = df[df['result'] == 0]['occupation'].value_counts()
over_val_c = df[df['result'] == 1]['occupation'].value_counts()

under_bars = [under_val_c.get(cat, default=0) for cat in categories]
over_bars = [over_val_c.get(cat, default=0) for cat in categories]

x_ticks = np.arange(cat_count)

plt.bar(x_ticks - bar_w / 2, under_bars, width=bar_w)
plt.bar(x_ticks + bar_w / 2, over_bars, width=bar_w)


plt.xticks([])
plt.legend(['under 50k', 'over 50k'])
plt.title('Occupation histogram per result group')
plt.tight_layout()

plt.savefig('occupation_hist_per_group.png')
plt.show()
