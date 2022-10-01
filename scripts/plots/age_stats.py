from project.scripts.data_preparation import df
import matplotlib.pyplot as plt

bin_edges = range(df['age'].min(), df['age'].max() + 1)

plt.figure()

plt.hist(df['age'], bins=bin_edges)

plt.title('Age histogram')
plt.xlabel('age')
plt.savefig('age_hist.png')
plt.show()

# per group plot
df.groupby('result')['age'].plot(kind='kde')

plt.title('Age density estimation per result group')
plt.xlabel('age')
plt.legend(['under 50k', 'over 50k'])
plt.xlim(df['age'].min() - 5,
         df['age'].max() + 5)
plt.yticks([])
plt.ylim(bottom=0)
plt.tight_layout()

plt.savefig('age_kde.png')
plt.show()
