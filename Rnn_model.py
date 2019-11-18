import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('train.csv',nrows = 2)
print(train_df.dtypes)