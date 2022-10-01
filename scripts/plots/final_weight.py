import matplotlib.pyplot as plt
from project.scripts.data_preparation import df_wo_nan

plt.figure()
plt.ticklabel_format(axis='x', scilimits=(0, 0))

df_wo_nan.groupby('result')['fnlwgt'].plot(kind='kde')
plt.xlim(-.01e6, .8e6)
plt.ylim(bottom=0)

plt.xlabel('fnlwgt')
plt.title("Final weight density estimation per result group")
plt.legend(['under 50k', 'over 50k'])
plt.yticks([])
plt.tight_layout()


plt.savefig('final_weight_kde.png')
plt.show()
