import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utility_functions as u_func
from sklearn.preprocessing import Normalizer
from keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau

sns.set()
import warnings

warnings.filterwarnings("ignore")

train_df = pd.read_csv('../train.csv')
print(train_df.shape)
print(train_df.head())

u_func.description_func(train_df)

# the dtypes of building_id, meter and meter_reading can be changed to uint16, uint8 and float32 respectively
train_df['building_id'] = train_df.building_id.astype(np.uint16)
train_df['meter'] = train_df.meter.astype(np.uint8)
train_df['meter_reading'] = train_df.meter_reading.astype(np.float32)
print(train_df.dtypes)

# train_df['year'] = train_df['timestamp'].str.split('-',1).str[0]
# print('train_df unique year ',train_df['year'].value_counts())
# sns.distplot(train_df[train_df['building_id'] == 0]['meter_reading'])
# plt.show()

# print(train_df[train_df['building_id'] == 0]['timestamp'].describe())
# sns.distplot((train_df['building_id']))
# plt.show()

# group by building_id
grouped = train_df.groupby('building_id')
print(len(grouped.groups.keys()))


g = grouped.get_group(0)


def data_preprocess(g):

    g.drop(['building_id'], axis=1, inplace=True)

    d_t = g['timestamp'].str.split(' ', 1, expand=True)
    g['date'] = d_t[0]
    g['time'] = d_t[1]
    g.drop(['timestamp'], axis=1, inplace=True)
    del d_t

    d_s = g['date'].str.split('-', 2, expand=True)
    g['month'] = d_s[1]
    g['day'] = d_s[2]
    g.drop(['date'], axis=1, inplace=True)
    del d_s

    t_s = g['time'].str.split(':', 1, expand=True)
    g['hour'] = t_s[0]
    g.drop(['time'], axis=1, inplace=True)
    del t_s

    x_t = g[g['month'] != '12']
    x_v = g[g['month'] == '12']

    # sns.distplot(x_t['meter_reading'])
    # sns.distplot(x_v['meter_reading'])
    # plt.show()

    x_t['month'] = x_t['month'].astype('category')
    x_t['month'] = x_t['month'].cat.codes
    x_t['day'] = x_t['day'].astype('category')
    x_t['day'] = x_t['day'].cat.codes
    x_t['hour'] = x_t['hour'].astype('category')
    x_t['hour'] = x_t['hour'].cat.codes

    x_v['month'] = x_v['month'].astype('category')
    x_v['month'] = x_v['month'].cat.codes
    x_v['day'] = x_v['day'].astype('category')
    x_v['day'] = x_v['day'].cat.codes
    x_v['hour'] = x_v['hour'].astype('category')
    x_v['hour'] = x_v['hour'].cat.codes

    y_t = x_t['meter_reading']
    x_t.drop(['meter_reading'], axis=1, inplace = True)

    y_v = x_v['meter_reading']
    x_v.drop(['meter_reading'], axis=1 , inplace=True)
    # print(x_t.tail())
    return np.array(x_t),np.array(x_v), y_t, y_v


x_t, x_v, y_t, y_v = data_preprocess(g)

#Normalize the data
# x_t = Normalizer().fit(x_t).transform(x_t)
# x_v = Normalizer().fit(x_v).transform(x_v)

x_t = np.expand_dims(x_t, 2)
x_v = np.expand_dims(x_v, 2)

print('x_t shape: ', x_t.shape)
print('x_v shape: ', x_v.shape)
print('y_t shape: ', y_t.shape)
print('y_v shape: ', y_v.shape)

rnn_model = u_func.RNN_model()

#set callbacks
cp = ModelCheckpoint('rnn_model_1.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)
rp = ReduceLROnPlateau(monitor='val_acc', patience=3, factor=0.1,mode='max')
csv = CSVLogger('rnn_model.csv', separator=',', append=False)
cb = [cp, rp, csv]

rnn_model.fit(x_t, y_t, epochs=10, batch_size=32, validation_data=(x_v, y_v), callbacks=cb)

pred = rnn_model.predict(x_v)
print(pred[:5])
print(y_v[:5])
print(pred.shape)

sns.distplot(y_v)
plt.show()
sns.distplot(pred)
plt.show()

# u_func.distribution_plots(grouped)

# build a model with input as (number of building_ids,len(meter)) and output as len(meter_reading)
# to create such model len(meter_reading should be same for all building id so check for it

# check for length of meter_reading in all building_ids
# u_func.meter_reading_length_plot(grouped,'train')
#

# # -------------------------------- Test data ----------------------------------
# test_df = pd.read_csv('../test.csv')
# print(test_df['building_id'].nunique())
#
# test_df['year'] = test_df['timestamp'].str.split('-', 1).str[0]
# # print('test_df unique year ',test_df['year'].value_counts())
#
# # test data includes 2017 and 2018 data so group 2017 and '18 data separatly
# print('year split done')
# grouped_2017 = test_df[test_df['year'] == '2017'].groupby('building_id')
# print(len(grouped_2017.groups.keys()))
#
# u_func.meter_reading_length_plot(grouped_2017,'2017 test')
#
# grouped_2018 = test_df[test_df['year'] == '2018'].groupby('building_id')
# print(len(grouped_2018.groups.keys()))
#
# u_func.meter_reading_length_plot(grouped_2018,'2018 test')
