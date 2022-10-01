from project.scripts.data_preparation import df
import matplotlib.pyplot as plt

bin_edges = range(df['education-num'].min(), df['education-num'].max() + 1)

plt.figure()

plt.hist(df['education-num'], bins=bin_edges)

plt.title('Education histogram')
plt.xlabel('education number')
plt.tight_layout()

plt.savefig('education_hist.png')
plt.show()

# per group plot
df.groupby('result')['education-num'].plot(kind='kde', bw_method=0.15)

plt.title('Education density estimation per result group')
plt.xlabel('education number')
plt.legend(['under 50k', 'over 50k'])
plt.yticks([])
plt.ylim(bottom=0)
plt.xlim(1, 17)
plt.tight_layout()

plt.savefig('education_kde.png')
plt.show()
