# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:57:18 2015

@author: p_cohen
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import random
from functools import partial
import os
sys.path.append("C:/Git_repos/ProductClasses")
import matplotlib.pyplot as plt
import data_describe as dd

############################ Functions ##################################
def prep_data(data_path, name, is_train=1):
    """ Prepare data for Kaggle Competition """
    # load data
    df = pd.read_csv(data_path + name)
    # create validation if it is train data
    if is_train:
        df = stratified_in_out_samp(df, 'target', .75)
    # recode target as numeric
    if is_train:
        df['target_num'] = df['target'].map(lambda x: x[-1:]).astype(int)
    return df
    
def list_feats(df, non_feats):
    """ Creates list of all modeling features """
    # Create list of all columns
    Xfeats = df.columns.values.tolist()
    # remove non modeling columns from feature list
    for col in non_feats:
        Xfeats.remove(col)
    return Xfeats

def stratified_in_out_samp(df, strat_var, split_rt, seed=42):
    """ 
    This function splits raw data into a train and validation sample,
    that is stratified on the selected variable
    """
    # set seed
    random.seed(seed)
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
   
def baseline_models(df, feats, models, target='target'):
    """
    This function applies a standard method for creating a baseline
    classification prediction
    """
    # Name train and val
    trn = df[df.is_val == 0].reset_index()
    # Fit models  
    for name, vals in models.iteritems():
        if vals['type'] == 'frst':
            # define forest
            mod = RandomForestClassifier(n_estimators=vals['prms'][0], 
                                         n_jobs=8,
                                         max_depth=vals['prms'][1])
        if vals['type'] == 'boost':
            # create boost specification
            mod = GradientBoostingClassifier(n_estimators=vals['prms'][0], 
                                             max_depth=vals['prms'][1],
                                             learning_rate=vals['prms'][2])
        # fit model
        mod.fit(trn[feats], trn[target].values)
        # store forest
        vals['model'] = mod
    return models
    
def evaluate_models(df, models, feats, target="target"):
    """ Evaluate models """
    # Name train and val
    df_val = df[df.is_val == 1].reset_index()
    # test each model
    for name, vals in models.iteritems():
        # create predictions
        preds = vals['model'].predict_proba(df_val[feats])
        # score predictions
        score = multiclass_log_loss(df_val[target]-1, preds)
        print str(name) + " has a score of " + str(score)
        # store the predictions and score
        vals['score'] = score
        vals['preds'] = preds
    return models
            
    # initialize best boost score
    best_models = {'model' : 'none', 'score' : 10000, 'preds' : 'none'}
    # create list of best models
    best_models = [best_frst, best_boost]    
    # determine sum of scores from best models
    best_score = 100
    for mod in best_models:
        best_score = min(best_score, mod['score'])
    # create scaled weights for each model (arbitrary combination)
    for mod in best_models:
        mod['weight'] = 1 / ((mod['score'] - best_score + .1)*10)
        print mod['weight']
    # create a blended score
    blend_preds = 0
    for mod in best_models:
        blend_preds += mod['weight']*mod['preds']
    blend_score = multiclass_log_loss(val[target]-1, blend_preds)
    print "Blended score is: " + str(blend_score)
    return best_models
  
def create_subm(models, df_test, feats, subm_path, nm):
    """ 
    Creates a kaggle submission from models, assumes Kaggle rescaling
    
    Inputs: 
    
    models - dictionary with models to use, (predictions and weights optional)
    
    df - test data
    
    feats - features to predict on
    
    subm_path - path to folder storing submissions
    
    nm - name of submission
    """
    # load sample submission
    sample_sub = pd.read_csv(subm_path + 'sampleSubmission.csv')
    # reset all non id columns to zero
    sample_sub.ix[:, 1:] = 0
    # create blended predictions
    for mod in models:
        # isolate test features
        testX = df_test[feats]
        # create weight predictions
        scaled_preds = mod['weight']*mod['model'].predict_proba(testX)
        # Create weighted average of models in sample sub
        sample_sub.ix[:, 1:] += scaled_preds
    sample_sub.ix[:, 0] = sample_sub.ix[:, 0].astype(int)
    # export submission
    sample_sub.to_csv(subm_path + nm + '.csv', index=False)      
 
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
PATH = "S:/03 Internal - Current/Kaggle/Otto Group Product Classification/Structured Data/"
PATH2 = "05 Data Documentation/"
SUBM_PATH = "03 Final Datasets/Submissions/"
OUTPATH = PATH + PATH2

# Define models to test
mods = {
'forst3' : {'prms' : [3000, None], 'model' : 'none', 'type' : 'frst'},
'boost2' : {'prms' : [300, 3, .14], 'model' : 'none', 'type' : 'boost'},
'boost4' : {'prms' : [1400, 1, .13], 'model' : 'none','type' : 'boost'},
'boost5' : {'prms' : [1400, 1, .14], 'model' : 'none', 'type' : 'boost'},
'boost6' : {'prms' : [1400, 1, .16], 'model' : 'none', 'type' : 'boost'}
}  

df_train = prep_data(PATH+PATH2+"../01 Raw Datasets/", "train.csv")
df_test = prep_data(PATH+PATH2+"../01 Raw Datasets/", "test.csv", is_train=0)
Xfeats = list_feats(df_train, ['id', 'target', 'is_val', 'target_num'])  
fit_models = baseline_models(df_train, Xfeats, mods, target='target_num')
evaluate_models(df_train, fit_models, Xfeats, target="target_num")
create_subm(best_models, df_test, Xfeats, PATH + SUBM_PATH, 'testing')
