# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:57:18 2015

@author: p_cohen
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from functools import partial

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
   
def num_describe(df):
    """ 
    This funciton takes a dataframe and evaluates percentiles, missings,
    outliers and unique values. Outputs a dataframe that stores results
    Note: df must be numeric only
    """
    # Create a gauranteed non-missing column to compare to other 
    df_output = df.describe().transpose().reset_index()
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
    return df_output

   
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
    if target:
        X = df
    else:
        y = df[target]
        X = df.ix[:, (df.columns.values != target)]
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
    # create correlation table
    df_correl = X_num.corr()
    

        
path = "S:/03 Internal - Current/Kaggle/Otto Group Product Classification" 
path2 = "/Structured Data/05 Data Documentation/" 
outpath = path + path2       
df = pd.read_csv(path + "/Structured Data/01 Raw Datasets/train.csv")

detailed_describe(df, outpath,  'Otto prediction', 'target')