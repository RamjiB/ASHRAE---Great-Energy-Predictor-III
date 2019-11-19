import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM
from keras.optimizers import Adam


def distribution_plots(grouped):
    keys = grouped.groups.keys()
    for i, b_id in enumerate(keys):
        g = grouped.get_group(b_id)
        try:
            f = sns.distplot(g['meter_reading']).set_title(str(b_id))
            f.get_figure().savefig('distribution_plots/building_id_' + str(b_id) + '.png')
            f.get_figure().clf()
        except ValueError:
            print('missed to save', b_id)

        if i % 100 == 0: print("saved {} images".format(i))

    print('Plots saved Successfully')


def meter_reading_length_plot(grouped, data='train'):
    l = []
    keys = list(grouped.groups.keys())
    for i, b_id in enumerate(keys):
        g = grouped.get_group(b_id)
        l.append(len(g['timestamp']))
    plt.plot(keys, l, 'r*')
    plt.title('timestamp length for '+data+' data')
    plt.xlabel('keys')
    plt.ylabel('length of timestamp')
    plt.show()


def description_func(df):
    desc_details = pd.DataFrame()
    for i,col in enumerate(df.columns):
        desc_details[col] = df[col].describe()
    desc_details.loc['dtype'] = df.dtypes
    print(desc_details)


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
    cat_codes = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5,
                 '07': 6, '08': 7, '09': 8, '10': 9, '11': 10, '12': 11}

    g['month'] = g['month'].astype('category')
    g['month'] = g['month'].cat.rename_categories(cat_codes)
    g['day'] = g['day'].astype('category')
    g['day'] = g['day'].cat.codes
    g['hour'] = g['hour'].astype('category')
    g['hour'] = g['hour'].cat.codes

    x_t = g[g['month'] != 11]
    x_v = pd.concat([g[g['month'] == 10], g[g['month'] == 11]])


    # sns.distplot(x_t['meter_reading'])
    # sns.distplot(x_v['meter_reading'])
    # plt.show()
    # print(x_t.tail())

    return x_t, x_v


def data_creation(d, dataset='valid'):
    X, Y = [], []
    if dataset == 'valid':
        x = d[d['month'] == 10]
        y = np.append(d[d['month'] == 10]['meter_reading'].iloc[-1],
                      d[d['month'] == 11]['meter_reading'])
        x.drop(['meter_reading'], axis=1, inplace=True)
        if len(y) <= 745:
            z_y = np.zeros((745 - len(y),))
            y = np.append((np.array(y)), z_y, axis=0)
        if len(x) <= 744:
            z_x = np.zeros((744 - len(x), 9))
            x = np.append((np.array(x)), z_x, axis=0)
        print('-----------------------------------------------------------------')
        print('X_valid shape: ', np.expand_dims(x, 0).shape)
        print('Y_valid shape: ', np.expand_dims(y, 0).shape)
        print('-----------------------------------------------------------------')
        return np.expand_dims(x, 0), np.expand_dims(y, 0)

    for i, month in enumerate(d.month.unique()):

        if i == 10:
            break
        x = d[d['month'] == i]
        y = np.append(d[d['month'] == i]['meter_reading'].iloc[-1],
                      d[d['month'] == i + 1]['meter_reading'])
        x.drop(['meter_reading'], axis=1, inplace=True)
        if len(y) <= 745:
            z_y = np.zeros((745 - len(y),))
            y = np.append((np.array(y)), z_y, axis=0)
        if len(x) <= 744:
            z_x = np.zeros((744 - len(x), 9))
            x = np.append((np.array(x)), z_x, axis=0)
        X.append(x)
        Y.append(y)
    X = np.reshape(X, (len(X), 744, 9))
    Y = np.reshape(Y, (len(Y), 745))
    #
    print('-----------------------------------------------------------------')
    print('X_train shape: ', X.shape)
    print('Y_train shape: ', Y.shape)
    print('-----------------------------------------------------------------')
    return X, Y


def rmsle(y_true,y_pred):
        y_true = K.log(y_true+1)
        y_pred = K.log(y_pred+1)
        return K.sqrt(K.mean(K.square(y_pred-y_true)))


def RNN_model():
    m = Sequential()
    m.add(SimpleRNN(25, input_shape=(9, 1)))
    m.add(Dense(128, activation='relu'))
    m.add(Dense(1, activation='relu'))
    m.summary()
    m.compile(optimizer=Adam(lr=0.01), loss=rmsle, metrics=['accuracy'])
    return m


def Dense_model():
    m = Sequential()
    m.add(Dense(16, input_shape=(9,)))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(16,activation='relu'))
    m.add(Dense(1, activation='relu'))
    m.summary()
    m.compile(optimizer=Adam(lr=0.01), loss=rmsle, metrics=['accuracy'])
    return m


def lstm_model(i_s = (744, 9)):
    model = Sequential()
    model.add(LSTM(25, activation='relu', return_sequences= True, input_shape=i_s))
    model.add(LSTM(25, activation='relu'))
    model.add(Dense(745,activation='relu'))
    model.compile(optimizer='adam',loss=rmsle,metrics=['accuracy'])
    return model