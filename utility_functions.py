import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense
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


def rmsle(y_true,y_pred):
        y_true = K.log(y_true+1)
        y_pred = K.log(y_pred+1)
        return K.sqrt(K.mean(K.square(y_pred-y_true)))


def RNN_model():
    m = Sequential()
    m.add(SimpleRNN(8, input_shape=(4, 1)))
    # m.add(Dense(16, activation='relu'))
    m.add(Dense(1, activation='relu'))
    m.summary()
    m.compile(optimizer=Adam(lr=0.001), loss=rmsle, metrics=['accuracy'])
    return m