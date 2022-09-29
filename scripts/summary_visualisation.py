import matplotlib.pyplot as plt
import numpy as np

from data_preparation import df_nominal


# rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
#
# ax.legend()
#
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)


def summary_hist(df, attr, group):
    print(attr)
    plt.figure()
#     fig, ax = plt.subplots()
    if df[attr].dtype == object:
        bin_counts = df.groupby(attr)[group].value_counts()
        categories = df[attr].unique()
        colors = ['r', 'b']
        for x, category in enumerate(categories):
            for res in (0, 1):
                idx = (category, res)
                if idx not in bin_counts.keys():
                    continue
                width = 0.2
                plt.bar(x + width * res, bin_counts[idx], width, label='under',
                        color=colors[res])
        plt.xticks(range(len(categories)), categories, rotation=-20)

    else:
        df.groupby(group)[attr].hist(alpha=0.5, bins=20)

    plt.xlabel(attr)
    plt.show()


columns = df_nominal.columns
columns = columns.drop('result')

for column in columns:
    summary_hist(df_nominal, column, 'result')
