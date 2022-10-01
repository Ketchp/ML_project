import matplotlib.pyplot as plt

from project.scripts.data_preparation import df_nominal, df_wo_nan


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
