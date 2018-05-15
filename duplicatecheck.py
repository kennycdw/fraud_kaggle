# This script creates a new feature in an attempt to identify duplicated datapoints, which is prevalent in the dataset

import pandas as pd

#day 9
frm = 144038905
to = 1849038905

#day 8
frm8 = 81038905
frm9 = 119038905

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }

full = train_df = pd.read_csv("train.csv", parse_dates=['click_time'],
                       dtype=dtypes, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])


train_df = pd.read_csv("train.csv", parse_dates=['click_time'], skiprows=range(1, frm), nrows=to - frm,
                       dtype=dtypes, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['minute'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
train_df['seconds'] = pd.to_datetime(train_df.click_time).dt.second.astype('uint8')


hey = train_df[train_df.duplicated(subset = ['ip','app','os','channel','device','day','hour','minute','seconds'],keep = False)]

train_df['duplicate'] = 0
train_df.loc[train_df.duplicated(subset = ['ip','app','os','channel','device','day','hour','minute','seconds'],keep = False),'duplicate']=1
