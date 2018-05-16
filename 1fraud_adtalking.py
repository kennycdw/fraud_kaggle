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
cores = 4
MISSING8 = 255
filename = "model"

frm = 80903891
to = 184903890

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }

print('Loading train data to pandas ...', frm, to)
#train_df = pd.read_csv("train.csv", parse_dates=['click_time'], skiprows=range(1, frm), nrows=to - frm,
   #                    dtype=dtypes, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

#len_train = len(train_df)

#DEBUG
train_df = pd.read_csv('train_sample.csv',parse_dates=['click_time'])
len_train = len(train_df)



predictors = ['app','device','channel','os','hour']


print('Loading test data to pandas')
#test_df = pd.read_csv("test_supplement.csv", parse_dates=['click_time'], dtype=dtypes,
#                      usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

##DEBUG
#test_df = pd.read_csv("test_supplement.csv", nrows=200000, parse_dates=['click_time'], dtype=dtypes,
 #                            usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
test_df = pd.read_csv("test.csv", parse_dates=['click_time'], dtype=dtypes,
                             usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

print('Syncing dataframes')

train_df['click_id'] = MISSING32
train_df['click_id'] = train_df.click_id.astype('uint32')
test_df['is_attributed'] = MISSING8
test_df['is_attributed'] = test_df.is_attributed.astype('uint8')
train_df=train_df.append(test_df)

del test_df
gc.collect()

# Creates pivot count feature for every group in count_group
# Function updates predictors, returns dataset with new features
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




# Creates an operator feature for a particular group (primary) by secondary
# Operator feature can be 'unique', 'mean' or 'var'
# Function updates predictors, returns dataset with new features
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




# Time to next click features
# The code is modified from kaggle discussion and uses less RAM than previous code
# Stores every value in each saved permutation (bijection) in the form of a hash list
# Updates dataframe and predictors
def feature_nextclick(dataset, primary):
    new_feature = 'nextclick_{}'.format('_'.join(primary))
    D = 2**26
    # Creating a unique string
    uniquestr = ""
    for i in primary:
        uniquestr = uniquestr + dataset[i].astype(str)
    dataset['category'] = (uniquestr).apply(hash) % D
    # Creating a list with values 3000000000 and D long. Note that D is < 2**32
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    dataset['epochtime'] = dataset['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    # Category here recalls the unique identifier, t recalls the time recorded from click_time (zip iterates over two lists)
    # Iterating through our dataset, we locate the column of identifier (click_buffer[category])
    # and deduct from CURRENT recorded time
    # Then we saved a specific column of click_buffer as time recorded.
    # Start backwards, saved values for future clicks first, then work down the list.
    for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - t)
        click_buffer[category] = t
    ##del (click_buffer)
    # We have to reverse back
    QQ = list(reversed(next_clicks))
    # Remove created features
    train_df.drop(['epochtime', 'category'], axis=1, inplace=True)
    # Change datatype and append feature into predictors
    train_df[new_feature] = pd.Series(QQ).astype('float32')
    predictors.append(new_feature)
    ##del QQ, next_clicks
    gc.collect()


### SETTING UP FEATURES
print("Creating features...")

train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

# Count features
count_group = [['ip'],
               ['ip','hour'],
               ['app','channel'],
               ['ip','device'],
               ['ip','app'],
               ['ip','app','os']
               ]
train_df = feature_count(train_df,count_group)
gc.collect()

# Cumcount features
ccount_group = [['ip','device','os'],
                ['ip'],
                ['ip','app','os']
                ]

#feature_cumcount(train_df,ccount_group,'channel',"uint32")
#gc.collect()

# Unique/mean/var features
#train_df = feature_operatorcount(train_df,['ip','device','os'],'app','unique','uint32')
#gc.collect()
#train_df = feature_operatorcount(train_df,['ip','app','channel'],'hour','mean','float32')
#gc.collect()
#train_df = feature_operatorcount(train_df,['ip','app','os'],'hour','var','float32')
#gc.collect()

# time to next click
feature_nextclick(train_df,['ip','app','device','os'])
gc.collect()
feature_nextclick(train_df,['ip','app'])
gc.collect()
# We have yet do drop 'click_time'

print('Feature Engineering Completed')
print('Printing Diagnostics...')

train_df.info()

target = 'is_attributed'
categorical = ['app','device','os','channel','hour']
print(predictors)

print('Splitting data...')
test_df = train_df[len_train:]
train_df = train_df[:len_train]

print('Creating validation set')
seed = 5
test_size = 0.10
train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=seed)

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

print('Temporarily remove test set and start training model')
test_df.to_pickle('test.pkl.gz')
del test_df
gc.collect()

print("Training model")
start_time = time.time()

early_stopping_rounds = 50
num_boost_round=1000
evals_results = {}
categorical_features=categorical

# General hyperparameters used from Kaggle
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
    'learning_rate': 0.10,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':200, # because training data is extremely unbalanced
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': cores,
    'verbose': 0,
}



print("preparing validation datasets")
lgbtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )

lgbvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )

print( train_df[predictors].head() )
print( train_df[target].head() )
print( val_df[predictors].head() )
print( val_df[target].head() )

del train_df
del val_df
gc.collect()

print(lgb_params)
bst = lgb.train(lgb_params,
                 lgbtrain,
                 valid_sets=[lgbtrain, lgbvalid],
                 valid_names=['train','valid'],
                 evals_result=evals_results,
                 num_boost_round=num_boost_round,
                 early_stopping_rounds=early_stopping_rounds,
                 verbose_eval=10,
                 feval=None)

print("Model Report")
print("bst.best_iteration: ", bst.best_iteration)
print('auc'+":", evals_results['valid']['auc'][bst.best_iteration-1])

print('[{}]: model training time'.format(time.time() - start_time))

print("Feature importance")
print(predictors)
print(bst.feature_importance())


#### LOAD TEST CSV####################################

print("Re-reading test data...")
test_df = pd.read_pickle('test.pkl.gz')

#ACTUAL
#ACTUALtest_df = pd.read_csv("test.csv", parse_dates=['click_time'], dtype=dtypes,
 #                            usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

#DEBUG
ACTUALtest_df = pd.read_csv("test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes,
                             usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

#Creates a new dataframe along with just the primal features. We aim to merge the actual test dataframe here
id_supplement = pd.DataFrame()
id_supplement['ip'] = test_df['ip']
id_supplement['app'] = test_df['app']
id_supplement['device'] = test_df['device']
id_supplement['os'] = test_df['os']
id_supplement['channel'] = test_df['channel']
id_supplement['click_time'] = test_df['click_time']
#Predicting results for test_supplement
id_supplement['is_attributed'] = bst.predict(test_df[predictors], num_iteration=bst.best_iteration)
# Arrange them in order and take the average prediction as predicted
id_supplement = id_supplement.groupby(['ip','app','device','os','channel','click_time'])['is_attributed'].\
                        agg(['mean']).reset_index().rename(index=str,columns=({'ip': 'ip','mean': 'is_attributed'}))

# Creating dataframe for actual prediction
sub = pd.DataFrame()
sub = ACTUALtest_df.merge(id_supplement, how='left', on=['ip', 'app', 'device', 'os', 'channel', 'click_time'])

print("del ip,app,device,os,channel,click_weekday")
del sub['ip']
del sub['app']
del sub['device']
del sub['os']
del sub['channel']
del sub['click_time']

print("writing...")
sub.to_csv(filename + '.csv', index=False, float_format='%.9f')
os.remove('test.pkl.gz')

print("done...")
print( sub.head(10) )
#######################################################



