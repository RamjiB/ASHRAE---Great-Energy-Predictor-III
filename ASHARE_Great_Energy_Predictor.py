import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utility_functions as u_func
from sklearn.preprocessing import Normalizer
from keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau
import warnings

pd.set_option('display.max_columns',None)
sns.set()
warnings.filterwarnings("ignore")

train_df = pd.read_csv('../train.csv')
print(train_df.shape)
print(train_df.head())
print(train_df.isna().sum())

u_func.description_func(train_df)
print('------------------------------------------------------------------------------')

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

# --------------------------------------------------------------------------------------------
building_metadata = pd.read_csv('../building_metadata.csv')
print('----------------------------------------------------------------')
print(building_metadata.shape)
print(building_metadata.head())
print('----------------------------------------------------------------')
print(building_metadata.isna().sum())
print('----------------------------------------------------------------')
u_func.description_func(building_metadata)

# merge building metadata and train_df based on building_id before that fill the nan values with zero on building
# metadata and change the dtypes of square_feet,year_built and floor_count to category and site_id,building_id and
# square_feet to uint8, uint16 and uint32

building_metadata.fillna(0, inplace=True)
building_metadata['primary_use'] = building_metadata['primary_use'].astype('category')
building_metadata['primary_use'] = building_metadata['primary_use'].cat.codes
building_metadata['year_built'] = building_metadata['year_built'].astype('category')
building_metadata['year_built'] = building_metadata['year_built'].cat.codes
building_metadata['floor_count'] = building_metadata['floor_count'].astype('category')
building_metadata['floor_count'] = building_metadata['floor_count'].cat.codes

building_metadata['site_id'] = building_metadata['site_id'].astype(np.uint8)
building_metadata['building_id'] = building_metadata['building_id'].astype(np.uint16)
building_metadata['square_feet'] = building_metadata['square_feet'].astype(np.uint32)

print('----------------------------------------------------------------')
u_func.description_func(building_metadata)

train_df = pd.merge(train_df,building_metadata,on=['building_id'],how='left')
print('----------------------------------------------------------------')
print(train_df.shape)
u_func.description_func(train_df)

# group by building_id
grouped = train_df.groupby('building_id')
print(len(grouped.groups.keys()))


g = grouped.get_group(0)


x_t, x_v = u_func.data_preprocess(g)
#
# #Normalize the data
# x_t = Normalizer().fit(x_t).transform(x_t)
# x_v = Normalizer().fit(x_v).transform(x_v)
# #
# # x_t = np.expand_dims(x_t, 2)
# # x_v = np.expand_dims(x_v, 2)
# #
print('----------------------------------------------------------------')
print('x_t shape: ', x_t.shape)
print('x_v shape: ', x_v.shape)
print('----------------------------------------------------------------')

# creating time series data
x_t, y_t = u_func.data_creation(x_t, dataset='train')
x_v, y_v = u_func.data_creation(x_v, dataset='valid')

# --------------------- Model ------------------------
#
# model = u_func.RNN_model()
# model = u_func.Dense_model()
model = u_func.lstm_model()
#
# # set callbacks
cp = ModelCheckpoint('lstm_model_1.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)
rp = ReduceLROnPlateau(monitor='val_acc', patience=3, factor=0.1,mode='max')
csv = CSVLogger('lstm_model.csv', separator=',', append=False)
cb = [cp, rp, csv]
#
model.fit(x_t, y_t, epochs=1, batch_size=32, validation_data=(x_v, y_v), callbacks=cb)
#
pred = model.predict(x_v)
print(pred[:5])
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
