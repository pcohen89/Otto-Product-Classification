# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:57:18 2015

@author: p_cohen
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from functools import partial

def compare_colcount(data1, data2):
    """ this compares the length of two data frames """
    return data1.count() - data2.count()
    
def pct_describe(data):
    """ acts as a wrapper for df.describe() """
    return data.describe() 
    
def outliers(data):
    """ this computes num observations 4stds from mean """
    mean = data.mean()
    std = data.std()
    return (np.abs(data - mean) > 4*std).sum()
    
def describe(X, outpath, name, target=""):
    """ This will export summaries of a data set
    
    Parameters
    -----------
    
    X : pandas Dataframe, shape = [n_samples, n_features]
    
    outpath :  path into directory where outputs will be stored
    
    name : name of dataset, to be used in naming outputs
    
    target : the name of a target variable. Defaults to no target
    specified, which will skip certain steps
        
    """
    
    # Print most basic features of the data 
    print "Obs in " + name + ": " + str(X.iloc[:, 0].count())
    print "Cols in " + name + ": " + str(X.iloc[0, :].count())
    # split data into numeric and non-numeric
    X_num = X._get_numeric_data()
    mask = X.isin(X_num).all(0)
    X_str = X.ix[:, ~mask]
    # print number of numeric columns
    print "Numeric cols in " + name + ": " + str(X_num.iloc[0, :].count())
    # column counts (df.count() doesn't count missings)
    X['one'] = 1
    # Create a gaurantee non-missing column to compare to other 
    df_output = X.describe().transpose().reset_index()

    # Store X.one as one of the arguments to compare_colcount
    miss_count = partial(compare_colcount, X.one)
    # Compute missings
    df_missings = pd.DataFrame(X.apply(miss_count, axis=0)).reset_index()
    df_missings.columns = ['feature', 'missings']
    df_output['missings'] = df_missings['missings']

    # Compute outliers
    X_num = X._get_numeric_data()
    df_outliers = pd.DataFrame(X_num.apply(outliers, axis=0)).reset_index()
    df_output = df_output.merge(df_outliers, how='outer')
    # rename index to 'column'
    out_cols = df_output.columns.values
    out_cols[0] = 'column'
    df_output.columns = out_cols

    
    # Compute feature types
    print X.dtypes
    
        
path =  "S:/03 Internal - Current/Kaggle/Otto Group Product Classification"           
df = pd.read_csv(path + "/Structured Data/01 Raw Datasets/train.csv")

describe(df, "", "")