# This script creates the aggregating features in a separate excel document, for easy merging later.
# Bayes features, pivot features, and time to next click features are considered here
# Ideas are inspired from kaggle discussion forum.

import pandas as pd
import numpy as np
import gc
import pytz
import datetime


dataset = pd.read_csv('train_sample.csv',parse_dates = ['click_time'])
dataset['hour'] = pd.to_datetime(dataset.click_time).dt.hour.astype('uint8')
dataset['day'] = pd.to_datetime(dataset.click_time).dt.day.astype('uint8')
dataset['minute'] = pd.to_datetime(dataset.click_time).dt.minute.astype('uint8')
dataset['second'] = pd.to_datetime(dataset.click_time).dt.second.astype('uint8')

countfeatures = [['hour'],
                 ['ip'],
                 ['app','channel'],
                 ['ip','app'],
                 ['ip','os'],
                 ['ip','app','os']
                 ]

#Creates pivoting count features as well as conditional probability
def bayesfunction(dataset,featuregroups):
    for group in featuregroups:
        #Creating feature name, alternatively use join
        variablename = ""
        for i in group:
            variablename = variablename + i + "_"
        variablename = variablename[:-1]
        print("Pivoting by ", variablename, '...')
        pivot = pd.pivot_table(dataset, index=group, values=['is_attributed'], aggfunc=[np.sum, len, np.mean])
        pivot = pivot.rename(index=str, columns={"len": "%s.count" % variablename, "": "%s" % variablename,
                                                 "mean": "%s.percentage" % variablename})
        pivot.columns = pivot.columns.droplevel(level=1)
        pivot = pivot.reset_index()
        del pivot['sum']
        print("Bayes of ", variablename, "feature completed!")
        pivot.to_csv('count_pivot_%s.csv' % variablename, index=False)

bayesfunction(dataset,countfeatures)
gc.collect()


#consider looking at frequency of time

def force_list(*arg):
    ''' Takes a list of arguments and returns the same, 
    but where all items were forced to a list.
    example : list_1,list_2=force_list(item1,item2)
    '''
    Gen=(x if isinstance(x,list) else [x] for x in arg)
    if len(arg)>1:
        return Gen
    else:
        return next(Gen)

#Consider
def get_window(my_data, grouped, aggregated, aggregators=['count'], dt=10, time_col='click_time'):
    ''' Returns a dataframe with the original index and rolling windows of dt minutes of the aggregated columns,
    per unique tuple of the grouped features, calculated with a given aggregator. The advantage of returning the new values only is that one
    can then join them as one wishes and in a pipeline.

        Args:
        data (pd.DataFrame): dataframe to add features to.
        grouping (list/string): column(s) to group by.
        aggregated (list): columns to aggregate on the rolling window.
        aggregators (list/string): methods to aggregate by.
        dt (int) : window size, in minutes
        time_col (datetime): column to use for the windows
    Returns:
        pd.DataFrame with similar index as input: rolling windows of the aggregated featured, relative to the grouped (i.e. fixed) ones.
    '''
    grouped, aggregated, aggregators = force_list(grouped, aggregated, aggregators)
    aggregation = {x: aggregators for x in aggregated}
    aggregation['click_id'] = lambda x: x[-1]

    dt = str(dt) + 'T'

    ''' Pandas Bug : multiple aggregates with groupby AND rolling is currently an open bug in Pandas 0.22, which is an issue as we need to keep the 'click_id' as key value here.
    new_frame=(my_data
               .groupby(grouped)[aggregated+['click_id']]
               .rolling(window=dt)
               .agg({'os':np.mean,'click_id':lambda x:x[-1]})
               .reset_index(level=0,drop=True)
               .add_suffix('_'.join(['',dt,agg_method]))
              )
    '''

    # Alternative approach, note that it will always return the biggest dtype, but we can't predict if float or int before hand.
    new_frame = (
        pd.concat([(grp
                    .set_index(time_col)
                    .rolling(window=dt)
                    .agg(aggregation)) for name, grp in my_data.groupby(grouped)[aggregated + ['click_id']]
                   ])
            .reset_index(drop=True)
            .set_index(('click_id', '<lambda>'))
            .sort_index()
    )

    # This may be unnecessary, but I have experienced merging issues before due to index names
    new_frame.index.name = None
    # example name 10T_os_count_by_ip
    new_frame.columns = ['{}T_'.format(dt) + '_'.join(list(col) + ['_by_'] + grouped) for col in new_frame.columns]
    return new_frame



# Group of features to consider
time_to_next_click = [ ['ip', 'app', 'device', 'os', 'channel'],
                         ['ip', 'app','os', 'device'], #Performs very strongly
                         ['ip', 'app']
                         ]

# Creates time to next click features and append them to dataset
def timefeaturefunction(dataset,featuregroups):
    for i in featuregroups:
        # Column name
        new_feature = 'nextClick_{}'.format('_'.join(i))
        # Select from this dataframe subset
        all_features = i + ['click_time']
        print(">> Grouping and creating ", new_feature, "time_feature")
        dataset[new_feature] = dataset[all_features].groupby(i).click_time.transform(
            lambda x: x.diff().shift(-1)).dt.total_seconds()
        # Just the time for now
        dataset[new_feature] = dataset[new_feature].fillna(value=99999999)

timefeaturefunction(dataset,time_to_next_click)

### Frequency of a particular instance (group/user) clicking on the same group (cumcount)

cumulativecount = [ ['ip', 'app', 'device', 'os', 'channel'],
                    ['ip', 'app'],
                    ['ip','app','channel']
                    ]


def cumulativecountfunction(dataset,featuregroups):
    for i in featuregroups:
        new_feature = 'cumcount_{}'.format('_'.join(i))
        print("Creating Cumulative", new_feature)
        dataset['reverse_' + new_feature] = dataset.groupby(i).cumcount()
        dataset['next_' + new_feature] = dataset.iloc[::-1].groupby(i).cumcount().iloc[::-1]

cumulativecountfunction(dataset,cumulativecount)

dataset.head()
