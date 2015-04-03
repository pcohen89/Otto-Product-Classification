# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:57:18 2015

@author: p_cohen
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from functools import partial
import os
sys.path.append("C:/Git_repos/ProductClasses")
import matplotlib.pyplot as plt
import data_describe as dd

############################ Functions ##################################
def stratified_in_out_samp(df, strat_var, split_rt):
    """ 
    This function splits raw data into a train and validation sample,
    that is stratified on the selected variable
    """
    # Group data by strat var
    grouped = df.groupby(df[strat_var])
    # create a frame to store is_val values
    df_validation = pd.DataFrame(columns=['id', 'is_val'])
    # Loop through groups
    for name, group in grouped:
        # count observations in group
        num_obs = len(group.index)
        # create a random digit for each observation in group
        group['rand'] = pd.Series(np.random.rand(num_obs), index=group.index)
        # encode is_val with observations selected by rand digit over thrshld 
        group['is_val'] = group.rand > group.rand.quantile(split_rt)
        # Store is_val values in validation
        df_output = group[['id', 'is_val']]
        df_validation = df_validation.append(df_output, ignore_index=True)
    # merge df_validation on df
    df = df.merge(df_validation, on='id')
    # return df
    return df
 
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/
    is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/
    12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota
   
def baseline_models(df, feats, target='target'):
    """
    This function applies a standard method for creating a baseline
    classification prediction
    """
    # Name train and val
    trn = df[df.is_val == 0].reset_index()
    val = df[df.is_val == 1].reset_index()
    # Define forests to test
    forests = {'forst1' : [3000, None],
               'forst2' : [30, 25],
               'forst3' : [30, 12]}
    forests = {'forst1' : [30, None],
               'forst2' : [30, 25],
               'forst3' : [30, 12]}
    # initialize best forest score
    best_frst = 10000
    # fit and evaluate each forest
    for name, params in forests.iteritems():
        # define forest
        frst = RandomForestClassifier(n_estimators=params[0], n_jobs=8,
                                      max_depth=params[1])
        # fit forest
        frst.fit(trn[feats], trn[target].values)
        # create predictions
        preds = frst.predict_proba(val[feats])
        # evaluate predictions
        score = multiclass_log_loss(val[target]-1, preds)
        print str(name) + " has a score of " + str(score)
        # store the predictions of the best forest run
        if score < best_frst:
            best_frst = score
            best_frst_preds = preds
    # initialize best boost score
    best_boost = 10000
    # define boosted trees to try
    boosts = {'boost1' : [300, 3, .11],
              'boost2' : [300, 3, .13],
              'boost3' : [300, 3, .14],
              'boost4' : [1200, 1, .07],
              'boost5' : [1200, 1, .08],
              'boost6' : [1200, 1, .09]}
              
    # fit and test each boost specification
    for name, params in boosts.iteritems():
        # create boost specification
        boost = GradientBoostingClassifier(n_estimators=params[0],
                                           max_depth=params[1],
                                           learning_rate=params[2])
        # fit boosted trees
        boost.fit(trn[feats], trn[target].values)
        # create predictions
        preds = boost.predict_proba(val[feats])
        # score predictions
        score = multiclass_log_loss(val[target]-1, preds)
        print str(name) + " has a score of " + str(score)
        # store the predictions of the best boost run
        if score < best_boost:
            best_boost = score
            best_boost_preds = preds
    blend = best_boost_preds + best_frst_preds
    score = multiclass_log_loss(val[target]-1,  blend)
    print "Blended score is: " + str(score)
     
            
 
############################ Tests ##################################
def test_strat_samp():
    """ 
    Tests that stratified_in_out_samp, returns an train and validation
    that are actually stratfied by the chosen var
    """
    # create data (duplicate vectore 1000 times, and create ID)
    d = {'target' : [1, 1, 1, 1, 1, 1, 1, 2]*1000, 'id' : range(8000)}
    df = pd.DataFrame(d)
    # Run function
    df_to_test = stratified_in_out_samp(df, 'target', .3)
    # create grouped version of df_to_test by target
    grouped = df_to_test.groupby(df.target)
    # Check that each level of target is sampled proportionally
    for name, group in grouped:
        if abs(group.is_val.mean()-.7) > .01:
            print group.is_val.mean()
            raise Exception('Sample not properly stratified')
    print('Function stratifies sample correctly')

########################### Live Code ###################################
PATH = "S:/03 Internal - Current/Kaggle/Otto Group Product Classification"
PATH2 = "/Structured Data/05 Data Documentation/"
OUTPATH = PATH + PATH2
df = pd.read_csv(PATH + "/Structured Data/01 Raw Datasets/train.csv")

df_wval = stratified_in_out_samp(df, 'target', .75)
df_wval['target_num'] = df_wval['target'].map(lambda x: x[-1:]).astype(int)
Xfeats = df_wval.columns.values.tolist()
non_feats = ['id', 'target', 'is_val', 'target_num']
for col in non_feats:
    Xfeats.remove(col)
baseline_models(df_wval, Xfeats, target='target_num')
