# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:57:18 2015

@author: p_cohen
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from functools import partial
import os
import matplotlib.pyplot as plt

############################ Functions ##################################
def split_num_str(df):
    """ This data seprates numeric columns from str columns """
    # get numeric data using buildin pandas method
    df_num = df._get_numeric_data()
    # create a condition for whether cells in df are in df_num
    mask = df.isin(df_num).all(0)
    # select all cells not in df_num for df_string
    df_str = df.ix[:, ~mask]
    return df_num, df_str

def pct_describe(df):
    """ acts as a wrapper for df.describe() """
    return df.describe()

def outliers(df):
    """ this computes number of obs 4stds from mean """
    mean = df.mean()
    std = df.std()
    return (np.abs(df - mean) > 4*std).sum()

def unique_vals(df):
    """ This counts the number of unique values of a df column """
    return len(df.drop_duplicates())

def rename_col_position(df, new_name, position):
    """ This renames the a dataframe column """
    # pulls the existing columns
    out_cols = df.columns.values
    # changes column in particular position number
    out_cols[position] = new_name
    # resaves the columns names
    df.columns = out_cols
    return df

def miss_count(df):
    """ Returns the number of missings """
    return len(df) - df.count()

def hicorrel(df, threshold=.7):
    """
    Used to evaluate a correlation matrix to see if values above threshold
    """
    # if column has no 1 value, raise error
    if len(df[df==1]) == 0:
        raise Exception("Not perfectly correlated with self?")
    # if column is perfectly correlated with a different column raise error
    elif len(df[df==1]) > 1:
        raise Exception("More than 1 perfect correlation")
    # Determine whether data column has any correlations above threshold
    df_abs = np.abs(df)
    df_abs_noself = df_abs[df_abs!=1]
    max_val = df_abs_noself.max()
    return max_val > threshold

def num_describe(df):
    """
    This function takes a dataframe and evaluates percentiles, missings,
    outliers and unique values. Outputs a dataframe that stores results
    Note: df must be numeric only
    """
    if len(df.columns) != len(df._get_numeric_data().columns):
        raise TypeError("Please, use only numeric data with num_describe")
    # Call standard pandas describe
    df_output = df.describe().transpose().reset_index()
    # Create dictionary of functions to apply to the data
    funcs = {'func1' : [miss_count, 'num_missings'],
             'func2' : [outliers, 'outliers_4std'],
             'func3' : [unique_vals, 'unique_values']}
    for key, value in funcs.iteritems():
        # Apply function to data
        df_new_stat = pd.DataFrame(df.apply(value[0], axis=0)).reset_index()
        # Rename column to describe statistic
        df_new_stat = rename_col_position(df_new_stat, value[1], 1)
        # Merge new stat onto output data set
        df_output = df_output.merge(df_new_stat, how='outer')
    # rename index to 'column'
    df_output = rename_col_position(df_output, 'column', 0)
    # drop 'one' column stats
    df_output = df_output[df_output.column != 'one']
    return df_output

def correl_describe(df, outpath, nm):
    """
    Calculates the correlation between numeric columns and outputs both
    a full correlation table and a small table that focuses on features with
    high correlations
    """
    if len(df.columns) != len(df._get_numeric_data().columns):
        raise TypeError("Please, use only numeric data with correl_describe")
    # create correlation table
    df_correl = df.corr()
    # store rows and columns with high correlation values
    df_hicorr_cols = df_correl.apply(hicorrel, axis=0)
    df_hicorr_rows = df_correl.apply(hicorrel, axis=1)
    # take subsample of corr table with only high corr columns and rows
    df_correl_zoomin = df_correl.ix[df_hicorr_rows, df_hicorr_cols]
    out_correl = outpath + "correlations/"
    # make corrrelations sub directory if one does not exist
    if not os.path.exists(out_correl):
        os.makedirs(out_correl)
    # save both raw and zoomed in correlations in sub directory
    df_correl.to_csv(out_correl + str(nm) + "all correl.csv", index=False)
    df_correl_zoomin.to_csv(out_correl + str(nm) + "high correl.csv",
                            index=False)

def draw_dists(series, path):
    """ Create distribution charts and save them """
    # For vars with less than 100 unique values, treat as categorical
    if len(series.drop_duplicates()) < 100:
        # Count frequency of each value
        clpsd_series = series.groupby(series).count()
        # create bar chart
        plt.bar(clpsd_series.index, clpsd_series.values, width=1)
    # for vals with more than 100 unique vals, treat as continuous
    else:
        # Create a historgram
        hist_min_max = (series.quantile(.01), series.quantile(.99))
        num_bins = min(len(series.drop_duplicates()), 50)
        plt.hist(series, bins=num_bins, range=hist_min_max)
    # Add titles
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(str(series.name))
    # Save chart
    plt.savefig(path + str(series.name) + ".png", format='png')
    # clear chartspace
    plt.close("all")

def chart_feats(df, outpath, name):
    """ Charts the distributions of numeric features """
    if len(df.columns) != len(df._get_numeric_data().columns):
        raise TypeError("Please, use only numeric data with chart_feats")
    # create path for distributions
    out_dist_pth = outpath + "distributions/"
    # make corrrelations sub directory if one does not exist
    if not os.path.exists(out_dist_pth):
        os.makedirs(out_dist_pth)
    # Bind the path to the draw dists function
    pathed_draw = partial(draw_dists, path=out_dist_pth)
    # Calculate and save pictures of feature distributions
    df.apply(pathed_draw, axis=0)

def detailed_describe(df, outpath, name, target=""):
    """ This will export summaries of a data set

    Parameters
    
    -----------

    df : pandas Dataframe, shape = [n_samples, n_features]

    outpath :  path into directory where outputs will be stored
    
    name : name of dataset, to be used in naming outputs

    target : the name of a target variable. Defaults to no target
    specified, which will skip certain steps

    """
    if target != "":
        y = df[target]
        X = df.ix[:, (df.columns.values != target)]
    else:
        X = df
    # Print most basic features of the data
    print "Obs in " + name + ": " + str(X.iloc[:, 0].count())
    print "Cols (excld target) in " + name + ": " + str(X.iloc[0, :].count())
    # split data into numeric and non-numeric
    X_num, X_str = split_num_str(X)
    # print number of numeric columns
    print "Numeric cols in " + name + ": " + str(X_num.iloc[0, :].count())
    # summarize numeric features
    df_output = num_describe(X_num)
    # export summary
    df_output.to_csv(outpath + name + ' numeric summary.csv', index=False)
    # Analyze correlations
    correl_describe(X_num, outpath, name)
    # Chart variable distributions
    chart_feats(X_num, outpath, name)

########################### Tests #######################################

def test_miss_count():
    """ Tests miss_count """
    d = {'data_w_missing' : pd.Series([1., 2., None])}
    df = pd.DataFrame(d)
    val_1missexpected = miss_count(df.data_w_missing)
    if val_1missexpected == 1:
        return "Miss_count finds exactly the number (1) of missings expected"
    elif val_1missexpected < 1:
        return "Miss_count failed to find the single missing value created"
    else:
        return "Miss_count found more than the single missing value created"

def test_outliers():
    """ test that outliers counts outliers correctly """
    d = {'data_w_1outlier' : pd.Series([1., 1., 1., 1., 1., 1., 1., 1.,
                                        1., 1., 1., 1., 1., 1., 1., 1.,
                                        1., 1., 1., 1., 1., 1., 1., 1.,
                                        1., 1., 1., 1., 1., 1., 1., 1.,
                                        1., 1., 1., 1., 1., 1., 1., 1.,
                                        15., -15.])}
    df = pd.DataFrame(d)
    val_1outlexpected = outliers(df.data_w_1outlier)
    if val_1outlexpected  == 2:
        return "Outliers finds exactly the number (2) of outliers expected"
    elif val_1outlexpected  < 2:
        return "Outliers failed to find the two outliers"
    else:
        return "Outliers found more than the two outliers"

def test_hicorrel():
    """
    Test that the hicorrel function (and the way it is used in detailed
    describe) works
    """
    d = {'feat1' : pd.Series([1., 1., 1., 1., 2., 2., 1., 1., 1., 1., 1.,]),
         'feat2' : pd.Series([1., 2., 2., 3., 3., 4., 4., 5., -5., 6., -6.,]),
         'feat3' : pd.Series([1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6.,]),
         'feat4' : pd.Series([1., 2., 2., 3., 3., 4., 4., 5., 5., 5.5, 6.5,])
         }
    df = pd.DataFrame(d)
    df_correl = df.corr()
    # store rows and columns with high correlation values
    df_hicorr_cols = df_correl.apply(hicorrel, axis=0)
    df_hicorr_rows = df_correl.apply(hicorrel, axis=1)
    # take subsample of corr table with only high corr columns and rows
    df_correl_zoomin = df_correl.ix[df_hicorr_rows, df_hicorr_cols]
    zoom_cols = df_correl_zoomin.columns.values
    if (zoom_cols[0], zoom_cols[1]) == ('feat3', 'feat4'):
        print "Hi correlation matrix formed as expected"
    else:
        print "Correlation is function is broken"

################## Example of usind detailed_describe ###################
PATH = "S:/03 Internal - Current/Kaggle/Otto Group Product Classification"
PATH2 = "/Structured Data/05 Data Documentation/"
OUTPATH = PATH + PATH2
df = pd.read_csv(PATH + "/Structured Data/01 Raw Datasets/train.csv")

detailed_describe(df, OUTPATH,  'Otto prediction', 'target')
