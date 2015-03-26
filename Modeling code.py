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

def split_num_str(df):
    """ This data seprates numeric columns from str columns """
    df_num = df._get_numeric_data()
    mask = df.isin(df_num).all(0)
    df_str = df.ix[:, ~mask]
    return df_num, df_str
    
def compare_colcount(df1, df2):
    """ this compares the length of two data frames """
    return df1.count() - df2.count()
    
def pct_describe(df):
    """ acts as a wrapper for df.describe() """
    return df.describe() 
    
def outliers(df):
    """ this computes num observations 4stds from mean """
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
    
def hicorrel(df, threshold=.7):
    """ 
    Used to evaluate a correlation matrix to see if values above threshold
    """
    # Overwrite perfect correlation with self
    if len(df[df==1])==1:
        df[df==1] = 0
    # if two columns are perfectly correlated, raise error
    else:
        raise Exception("Two perfectly correlated columns")
    df_abs = np.abs(df)
    max_val = df_abs.max()
    return max_val > threshold

def num_describe(df):
    """ 
    This function takes a dataframe and evaluates percentiles, missings,
    outliers and unique values. Outputs a dataframe that stores results
    Note: df must be numeric only
    """
    df['one'] = 1
    # Create a gauranteed non-missing column to compare to other 
    df_output = df.describe().transpose().reset_index()
    # create a one column
    # Store X.one as one of the arguments to compare_colcount
    miss_count = partial(compare_colcount, df.one)
    # Create vector of fucntions to apply
    funcs = {'func1' : [miss_count, 'num_missings'],
             'func2' : [outliers, 'outliers_4std'],
             'func3' : [unique_vals, 'unique_values']}
    for key, value in funcs.iteritems(): 
        # Compute values
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
    """ Create charts and save them """
    plt.hist(series, 10)
    plt.show()
    
     
def chart_feats(df, outpath, name):
    """ Charts the distributions of numeric features """
    if len(df.columns) != len(df._get_numeric_data().columns):
        raise TypeError("Please, use only numeric data with chart_feats")
    # create path for distributions
    out_dist_pth = outpath + "distributions/"
    # make corrrelations sub directory if one does not exist
    if not os.path.exists(out_dist_pth):
        os.makedirs(out_dist_pth)
    
        
    
    
  
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
    print "Cols in " + name + ": " + str(X.iloc[0, :].count())
    # split data into numeric and non-numeric
    X_num, X_str = split_num_str(X)
    # print number of numeric columns
    print "Numeric cols in " + name + ": " + str(X_num.iloc[0, :].count())
    # summarize numeric features
    df_output = num_describe(X_num)
    # export summary
    df_output.to_csv(outpath + name + ' numeric summary.csv', index=False)
    # Analyze correlations
    correl_describe(df, outpath, name)
    # Chart variable distributions
    chart_feats(df, outpath, name)
   
path = "S:/03 Internal - Current/Kaggle/Otto Group Product Classification" 
path2 = "/Structured Data/05 Data Documentation/" 
outpath = path + path2       
df = pd.read_csv(path + "/Structured Data/01 Raw Datasets/train.csv")

detailed_describe(df, outpath,  'Otto prediction', 'target')
draw_dists(df.feat_90, "")
plt.hist(df.feat_90, 10)
plt.show()
