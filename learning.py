import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
pd.set_option('display.max_columns', None)
#paths

train_path = 'train.csv'
test_path = 'test.csv'
building_metadata_path = 'building_metadata.csv'
train_weather = 'weather_train.csv'
test_weather = 'weather_test.csv'

seed_no = 12

# data_preparation
# there are two files which needs to be merged with train data. Building_metadata which has
# information about buildings and other is weather data which has information like dew,temp,etc.,

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

building_meta_df = pd.read_csv(building_metadata_path)

# In building meta_data primary use column has categorical string values so encode it with
# labels suing sklearn

le = LabelEncoder()
building_meta_df['primary_use'] = le.fit_transform(building_meta_df['primary_use'])
building_meta_df.drop(['floor_count'],axis =1,inplace=True)

weather_train_df = pd.read_csv(train_weather)
weather_test_df = pd.read_csv(test_weather)

print(weather_train_df.isna().sum()/len(weather_train_df))

# why dropping the below columns?
weather_train_df.drop(['sea_level_pressure', 'wind_direction', 'wind_speed'], axis=1, inplace=True)
weather_test_df.drop(['sea_level_pressure', 'wind_direction', 'wind_speed'], axis=1, inplace=True)

weather_train_df = weather_train_df.groupby('site_id').apply(lambda g: g.interpolate(limit_direction='both'))
weather_test_df = weather_test_df.groupby('site_id').apply(lambda g: g.interpolate(limit_direction='both'))

# merging buliding metadata and train data
train_df = train_df.merge(building_meta_df, on="building_id")

# merging weather data with train data
train_df = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

# why removing the following details ?
train_df = train_df[~((train_df.site_id == 0) & (train_df.meter == 0) &
                    (train_df.building_id <= 104) & (train_df.timestamp < "2016-05.21"))]
train_df.reset_index(drop=True, inplace=True)

train_df.timestamp = pd.to_datetime(train_df.timestamp, format='%Y-%m-%d %H:%M:%S')
train_df['log_meter_reading'] = np.log1p(train_df.meter_reading)

test_df = test_df.merge(building_meta_df, on='building_id')
test_df = test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how="left")
test_df.reset_index(drop=True, inplace=True)
test_df.timestamp = pd.to_datetime(test_df.timestamp, format='%Y-%m-%d %H:%M:%S')

del building_meta_df,le
print(gc.collect())


def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                else:
                    if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype("category")

            end_mem = df.memory_usage().sum() / 1024 ** 2
            print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
            print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

            return df


train_df = reduce_mem_usage(train_df, use_float16=True)
test_df = reduce_mem_usage(test_df, use_float16=True)

weather_train_df.timestamp = pd.to_datetime(weather_train_df.timestamp, format='%Y-%m-%d %H:%M:%S')
weather_test_df.timestamp = pd.to_datetime(weather_test_df.timestamp,format='%Y-%m-%d %H:%M:%S')
weather_train_df = reduce_mem_usage(weather_train_df, use_float16=True)
weather_test_df = reduce_mem_usage(weather_test_df, use_float16=True)

print('------------------------------ time Feature Engineering ----------------------')

# time FEATURE Engineering

train_df['hour'] = train_df.timestamp.dt.hour
train_df['weekday'] = train_df.timestamp.dt.weekday
train_df['is_weekend'] = 1*(train_df['weekday']//5 == 1)
train_df['is_business_hr'] = 1* (train_df['hour'].between(8,15) == True)

test_df['hour'] = test_df.timestamp.dt.hour
test_df['weekday'] = test_df.timestamp.dt.weekday
test_df['is_weekend'] = 1*(test_df['weekday']//5 == 1)
test_df['is_business_hr'] = 1* (test_df['hour'].between(8,15) == True)

print('------------------------- Aggregate feature engineering ---------------------')
# Creating aggregate features for buildings at various levels

building_meter_df = train_df.groupby(['building_id', 'meter']).agg(
    mean_building_meter=("log_meter_reading", "mean"),
    median_building_meter=("log_meter_reading", "median")).reset_index()

train_df = train_df.merge(building_meter_df, on=["building_id", "meter"])
test_df = test_df.merge(building_meter_df, on=["building_id", "meter"])

building_meter_hr_df = train_df.groupby(["building_id", 'meter', 'hour']).agg(
    mean_building_meter=('log_meter_reading', 'mean'),
    median_building_meter=('log_meter_reading', 'median')
).reset_index()

train_df = train_df.merge(building_meter_hr_df,
                          on=['building_id', 'meter', 'hour'])
test_df = test_df.merge(building_meter_hr_df,
                        on=['building_id', 'meter', 'hour'])

building_meter_weekend_df = train_df.groupby(["building_id", 'meter', 'is_weekend']).agg(
    mean_building_meter=('log_meter_reading', 'mean'),
    median_building_meter=('log_meter_reading', 'median')
).reset_index()

train_df = train_df.merge(building_meter_weekend_df,
                          on=['building_id', 'meter', 'is_weekend'])
test_df = test_df.merge(building_meter_weekend_df,
                        on=['building_id', 'meter', 'is_weekend'])

building_meter_business_hr_df = train_df.groupby(["building_id", 'meter', 'is_business_hr']).agg(
    mean_building_meter=('log_meter_reading', 'mean'),
    median_building_meter=('log_meter_reading', 'median')
).reset_index()

train_df = train_df.merge(building_meter_business_hr_df,
                          on=['building_id', 'meter', 'is_business_hr'])
test_df = test_df.merge(building_meter_business_hr_df,
                        on=['building_id', 'meter', 'is_business_hr'])



print('--------------------------- Lags based Feature Engineering ----------------')


def create_lag_features(df, window):
    """
    Creating lag-based features looking back in time.
    """

    feature_cols = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"]
    df_site = df.groupby("site_id")

    df_rolled = df_site[feature_cols].rolling(window=window, min_periods=0)

    df_mean = df_rolled.mean().reset_index().astype(np.float16)
    df_median = df_rolled.median().reset_index().astype(np.float16)
    df_min = df_rolled.min().reset_index().astype(np.float16)
    df_max = df_rolled.max().reset_index().astype(np.float16)
    df_std = df_rolled.std().reset_index().astype(np.float16)
    df_skew = df_rolled.skew().reset_index().astype(np.float16)

    for feature in feature_cols:
        df[f"{feature}_mean_lag{window}"] = df_mean[feature]
        df[f"{feature}_median_lag{window}"] = df_median[feature]
        df[f"{feature}_min_lag{window}"] = df_min[feature]
        df[f"{feature}_max_lag{window}"] = df_max[feature]
        df[f"{feature}_std_lag{window}"] = df_std[feature]
        df[f"{feature}_skew_lag{window}"] = df_std[feature]

    return df


print('-------------- Creating features ------------------------')

weather_train_df = create_lag_features(weather_train_df, 18)
weather_train_df.drop(['air_temperature', 'cloud_coverage',
                       'dew_temperature', 'precip_depth_1_hr'], axis=1, inplace=True)

train_df = train_df.merge(weather_train_df,on=['site_id', 'timestamp'], how="left")

#print('--------------- convert test and train to csv --------------------------------------')
#train_df.to_csv('learning_train_df.csv',index = False)
#test_df.to_csv('learning_test_df.csv',index = False)

del weather_train_df
print(gc.collect())
#train_df = pd.read_csv('learning_train_df.csv')
#test_df = pd.read_csv('learning_test_df.csv')
#test_df = reduce_mem_usage(test_df,use_float16=True)

cat_features = ['building_id', 'primary_use', 'meter', 'weekday', 'hour']
all_features = [col for col in train_df.columns if col not in ['timestamp', 'site_id',
                                                               'meter_reading', 'log_meter_reading']]

# KFold cross validation
cv = 2
models = {}
cv_scores = {'site_id': [], 'cv_score': []}
for site_id in tqdm(range(16), desc="site_id"):
    print(cv, "fold CV for site_id:", site_id)
    kf = KFold(n_splits=cv, random_state=seed_no)
    models[site_id] = []
    x_train_site = train_df[train_df.site_id == site_id].reset_index(drop=True)
    y_train_site = x_train_site.log_meter_reading
    y_pred_train_site = np.zeros(len(x_train_site))

    score = 0

    for fold,(trn_idx,val_idx) in enumerate(kf.split(x_train_site,y_train_site)):
        x_tr, x_va = x_train_site.loc[trn_idx, all_features], x_train_site.loc[val_idx, all_features]
        y_tr, y_va = y_train_site.iloc[trn_idx], y_train_site.iloc[val_idx]

        dtrain = lgb.Dataset(x_tr, label=y_tr, categorical_feature=cat_features)
        dvalid = lgb.Dataset(x_va, label=y_va, categorical_feature=cat_features)

        watchlist = [dtrain, dvalid]
        params = {'objective': 'regression',
                  'num_leaves': 41,
                  'learning_rate': 0.049,
                  'bagging_freq': 5,
                  'bagging_fraction': 0.51,
                  'feature_fraction':0.81,
                  'metric':'rmse'}

        lgb_model = lgb.train(params,train_set=dtrain,num_boost_round=1000,valid_sets=watchlist,
                              verbose_eval=101,early_stopping_rounds=21)
        models[site_id].append(lgb_model)

        y_pred_val = lgb_model.predict(x_va, num_iteration=lgb_model.best_iteration)
        y_pred_train_site[val_idx] = y_pred_val

        rmse = np.sqrt((mean_squared_error(y_va,y_pred_val)))
        print('site_id: ', site_id, ", Fold: ", fold+1, ", RMSE: ", rmse)
        score += rmse/cv

        gc.collect()
    cv_scores['site_id'].append(site_id)
    cv_scores['cv_score'].append(score)

    print('\nSite Id: ', site_id, ", CV RMSE: ",
          np.sqrt(mean_squared_error(y_train_site, y_pred_train_site)), "\n")
print('--------------------------- CV scores ----------------------')
print(pd.DataFrame.from_dict(cv_scores))
pd.DataFrame.from_dict(cv_scores).to_csv('training_score_we_bh.csv',index = False)

del train_df, x_train_site, y_train_site, x_tr, y_tr, dtrain, x_va, y_va, dvalid, y_pred_train_site, y_pred_val, rmse, score, cv_scores
gc.collect()



# scoring on test data
print('---------------------- Test data ---------------------------------')

weather_test_df = create_lag_features(weather_test_df,18)
weather_test_df.drop(['air_temperature', 'cloud_coverage',
                       'dew_temperature', 'precip_depth_1_hr'], axis=1, inplace=True)

test_sites_df = []

for site_id in tqdm(range(16), desc='site_id'):
    print(' Preparing test data for site_id', site_id)
    x_test_site = test_df[test_df.site_id == site_id]
    weather_test_site = weather_test_df[weather_test_df.site_id == site_id]

    x_test_site = x_test_site.merge(weather_test_site, on=['site_id', 'timestamp'], how='left')
    row_ids_site = x_test_site.row_id
    x_test_site = x_test_site[all_features]
    y_pred_test_site = np.zeros(len(x_test_site))

    print("Scoreing for site_id", site_id)
    for fold in range(cv):
        lgb_model = models[site_id][fold]
        y_pred_test_site += lgb_model.predict(x_test_site, num_iteration=lgb_model.best_iteration) / cv
        gc.collect()

    test_site_df = pd.DataFrame({"row_id":row_ids_site,"meter_reading":y_pred_test_site})
    test_sites_df.append(test_site_df)

    print("scoring for site_id ", site_id, "completed\n")
    gc.collect()

# Submission
print('---------------------------------submission-------------------------')
sub = pd.concat(test_sites_df)
sub.meter_reading = np.clip(np.expm1(sub.meter_reading), 0, a_max=None)
sub.to_csv('learning_submission_we_bh.csv', index=False)
