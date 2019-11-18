import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from progress.bar import Bar
from utility_functions import distribution_plots
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('../train.csv')
print(train_df.dtypes)
print(train_df.shape)
print(train_df.head())

print(train_df['building_id'].nunique())
# sns.distplot(train_df[train_df['building_id'] == 0]['meter_reading'])
# plt.show()

print(train_df[train_df['building_id'] == 0]['timestamp'].describe())
# sns.distplot((train_df['building_id']))
# plt.show()

# group by building_id
grouped = train_df.groupby('building_id')
print(len(grouped.groups.keys()))
# distribution_plots(grouped)
