import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action="ignore",category=warnings)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

netflix_data = pd.read_csv("dataset/netflix_titles.csv")
df = netflix_data.copy()
df.head()
df.shape
df.columns
df.isnull().sum()
df.nunique()
data = df

data = data.dropna()
data.shape

#EDA 
for check_eda(dataframe) :
print(15*#)
      




