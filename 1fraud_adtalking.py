import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

MISSING32 = 999999999
val_size= 2500000
keeptrack = "01.40am"
cores = 4
MISSING8 = 255

frm = 144
to = 144

print('Loading train data to pandas ...', frm, to)
train_df = pd.read_csv("train.csv", parse_dates=['click_time'], skiprows=range(1, frm), nrows=to - frm,
                       dtype=dtypes, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

train_df = pd.read_csv('train_sample.csv')

predictors = []


# Creates pivot count feature for every group in count_group
# Function updates predictors, returns dataset
def feature_count(dataset,featuregroups):
    for i in featuregroups:
        new_feature = 'count_{}'.format('_'.join(i))
        print("Creating Count", new_feature)
        pivot = dataset[i].groupby(i).size().rename(new_feature).to_frame().reset_index()
        dataset = dataset.merge(pivot, on=i, how='left')
        del pivot
        gc.collect()
        predictors.append(new_feature)
    print('Count features done!')
    return (dataset)

count_group = [['ip'],
               ['app','channel']
               ]

train_df = feature_count(train_df,count_group)


# Creates an operator feature for a particular group (primary) by secondary
# Operator feature can be 'unique', 'mean' or 'var'
def feature_operatorcount(dataset, primary, secondary, operator, datatype):
    new_feature = operator + '_{}'.format('_'.join(primary)) + '_by_' + secondary
    print("Creating", operator ,"Count", new_feature)
    if operator == "unique":
        pivot = dataset[primary + [secondary]].groupby(primary)[secondary].nunique().reset_index().rename(columns={secondary: new_feature})
    elif operator == "mean":
        pivot = dataset[primary + [secondary]].groupby(primary)[secondary].mean().reset_index().rename(columns={secondary: new_feature})
    elif operator == "var":
        pivot = dataset[primary + [secondary]].groupby(primary)[secondary].var().reset_index().rename(columns={secondary: new_feature})
    dataset = dataset.merge(pivot, on = primary, how = 'left')
    del pivot
    dataset[new_feature]= dataset[new_feature].astype(datatype)
    gc.collect()
    predictors.append(new_feature)
    print(operator,'Count', new_feature, "done!")
    return (dataset)

train_df = feature_operatorcount(train_df,['ip','device','os'],'app','mean','uint32')
train_df = feature_operatorcount(train_df,['ip','device','os'],'app','unique','uint32')
train_df = feature_operatorcount(train_df,['ip','device','os'],'app','var','uint32')

### Frequency of a particular instance (group/user) clicking on the same group (cumcount).
# We will create both a forward count and a backward count for each instance
# Updates dataframe and predictors
def feature_cumcount(dataset, featuregroups, secondary, datatype):
    #secondary does not matter in this feature creating but I added this for consistency in all functions
    for i in featuregroups:
        new_feature = 'cumcount_{}'.format('_'.join(i))
        print("Creating Cumulative", new_feature)
        temp = dataset[i + [secondary]].groupby(i)[secondary].cumcount()
        dataset['CC_' + new_feature] = temp.values
        dataset['CC_' + new_feature].astype(datatype)
        del temp
        tempR = dataset[i + [secondary]].iloc[::-1].groupby(i)[secondary].cumcount().iloc[::-1]
        dataset['CCreverse_' + new_feature] = tempR.values
        del tempR
        predictors.append('CC_' + new_feature)
        predictors.append('CCreverse_' + new_feature)
        gc.collect()
    print('Cumcount features done!')


#Cumcount feature
ccount_group = [['ip','device','os'],
                ['ip'],
                ['ip','app','os']
                ]

feature_cumcount(train_df,ccount_group,'channel',"uint32")
