import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utility_functions as u_func

sns.set()
import warnings

warnings.filterwarnings("ignore")

# train_df = pd.read_csv('../train.csv')
# print(train_df.dtypes)
# print(train_df.shape)
# print(train_df.head())
#
# print(train_df['building_id'].nunique())
# train_df['year'] = train_df['timestamp'].str.split('-',1).str[0]
# print('train_df unique year ',train_df['year'].value_counts())
# sns.distplot(train_df[train_df['building_id'] == 0]['meter_reading'])
# plt.show()

# print(train_df[train_df['building_id'] == 0]['timestamp'].describe())
# sns.distplot((train_df['building_id']))
# plt.show()

# group by building_id
# grouped = train_df.groupby('building_id')
# print(len(grouped.groups.keys()))


# u_func.distribution_plots(grouped)

# build a model with input as (number of building_ids,len(meter)) and output as len(meter_reading)
# to create such model len(meter_reading should be same for all building id so check for it

# check for length of meter_reading in all building_ids
# u_func.meter_reading_length_plot(grouped,'train')
#

# -------------------------------- Test data ----------------------------------
test_df = pd.read_csv('../test.csv')
print(test_df['building_id'].nunique())

test_df['year'] = test_df['timestamp'].str.split('-', 1).str[0]
# print('test_df unique year ',test_df['year'].value_counts())

# test data includes 2017 and 2018 data so group 2017 and '18 data separatly
print('year split done')
grouped_2017 = test_df[test_df['year'] == '2017'].groupby('building_id')
print(len(grouped_2017.groups.keys()))

u_func.meter_reading_length_plot(grouped_2017,'2017 test')

grouped_2018 = test_df[test_df['year'] == '2018'].groupby('building_id')
print(len(grouped_2018.groups.keys()))

u_func.meter_reading_length_plot(grouped_2018,'2018 test')
