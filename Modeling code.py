# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:57:18 2015

@author: p_cohen
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import sys
sys.path.append('../python/')
import xgboost as xgb
import pandas as pd
import numpy as np
import random
import time
import copy
from sklearn.feature_selection import chi2, SelectPercentile, f_classif
import xgboost as xgb
from functools import partial
import os
sys.path.append("C:/Git_repos/compiled_xgboost/lib")
import xgboost as xgb
import data_describe as dd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

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
        # Uses last digit which is the numeric part of target
        df['target_num'] = df['target'].map(lambda x: x[-1:]).astype(int)
    return df
    
def build_Otto_vars(df):
    """ Build variables for Otto """
    # Create list of original, on-id vars
    orig_feats = list_feats(df)
    is_zero_nms = list_feats(df)
    # Create a row sum
    df['feat_row_sum'] = df.ix[:, orig_feats].sum(1)
    # Create a row std
    df['feat_row_std'] = df.ix[:, orig_feats].std(1)
    # Create is_zero binaries
    for index in range(0, len(is_zero_nms)):
        is_zero_nms[index] = "is_zero_" + is_zero_nms[index]
    df[is_zero_nms] = df[orig_feats] > 0
    # Count times non-zero
    df['feat_cnt_non_zero'] = df.ix[:, is_zero_nms].sum(1)
    # Clear mem space
    for name in is_zero_nms:
        del df[name]
    # Normalize original vars
    f = lambda x: (x-x.mean())/x.std()
    df[list_feats(df)] = df[list_feats(df)].apply(f, axis=0)
    # create column names for one hot encoded data
    return df

def list_feats(df):
    """ Creates list of all modeling features """
    # Create list of original, non-id vars
    orig_feats = list(df.columns.values)
    chosen_feats = list(df.columns.values)
    for name in orig_feats:
        if name[:4] != "feat":
            chosen_feats.remove(name)
    return chosen_feats
   
def select_feats(df, target="target"):
    """ Creates list of all modeling features """
    # Create list of original, non-id vars
    feats = list_feats(df)
    chosen_feats = list_feats(df)
    # Name train rows
    trn = df[df.is_val == 0].reset_index()
    # save feats chosen by forest
    selector = SelectPercentile(f_classif, percentile=97)
    selector.fit(trn[feats], trn[target].values)
    for i in range(0, len(selector.get_support())):
        if ~selector.get_support()[i]:
            chosen_feats.remove(feats[i])
    return chosen_feats

def stratified_in_out_samp(df, strat_var, split_rt, seed=20):
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
    # Name train rows
    trn = df[df.is_val == 0].reset_index()
    # xgb train
    xg_train = xgb.DMatrix(trn[feats], label=trn[target].values)
    # Fit models  
    for name, vals in models.iteritems():
        print "Fitting " + str(name)
        if vals['type'] == 'frst':
            # define forest
            mod = RandomForestClassifier(n_estimators=vals['prms'][0], 
                                         n_jobs=8, max_depth=vals['prms'][1])
        if vals['type'] == 'boost':
            # create boost specification
            mod = GradientBoostingClassifier(n_estimators=vals['prms'][0], 
                      max_depth=vals['prms'][1], learning_rate=vals['prms'][2], 
                      subsample=.8, max_features=80)
        if vals['type'] == 'svm':
            mod = SVC(C=vals['prms'][0], kernel=vals['prms'][1],
                      probability=True)
        # fit model
        t0 = time.time() 
        if vals['type'] != 'svm':                                    
            mod.fit(trn[feats], trn[target].values)
        else:
            # Subset the insample
            num_obs = len(trn.index)
            # create a random digit for each observation in group
            trn['rand'] = pd.Series(np.random.rand(num_obs), index=trn.index)
            svm_trn = pd.DataFrame(trn[trn.rand<.9].reset_index())
            print "Running svm with " + str(len(svm_trn.index)) + " obs"  
            # Fit model                             
            mod.fit(svm_trn[feats], svm_trn[target].values)
        title = "It took {time} minutes to run " + name
        print title.format(time=(time.time()-t0)/60)
        #else:
        #    mod = bst
        # store forest
        vals['model'] = mod
    return models
    
def evaluate_models(df, models, feats, encoder=None, target="target"):
    """ Evaluate models """
    # Name train and val
    df_val = df[df.is_val == 1].reset_index()
    # keep best model of each type
    best_mods = {}
    # test each model
    for name, vals in models.iteritems():
        if vals['type'] != 'xbm':
            # create predictions
            preds = vals['model'].predict_proba(df_val.ix[:,feats])
        else:
            # create predictions
            preds = vals['model'].predict(df_val.ix[:,feats])
        # score predictions
        score = multiclass_log_loss(df_val[target]-1, preds)
        print str(name) + " has a score of " + str(score)
        # store the predictions and score
        vals['score'] = score
        vals['preds'] = preds
        # if a model of the current type is already stored in the best_mods
        if name in best_mods:
            # if the score of the current model is at least close in score
            # to the best model
            if .75 * vals['score'] < best_mods[vals['type']]['score']:
                # replace old 'best' model with current model
                best_mods[name] = vals
        # if no model of current type is in best_mods
        else:
            best_mods[name] = vals
    return best_mods

def eval_blend_best(df_train, best_mods, target='target'):          
    """ Creates a niave blend of the best models and evalutes it """
    # Name train and val
    df_val = df_train[df_train.is_val == 1].reset_index()
    # determine sum of scores from best models
    best_score = 100
    # determine number of models
    num_mods = len(best_mods.keys())
    for name, vals in best_mods.iteritems():
        best_score = min(best_score, vals['score'])
    # create scaled weights for each model (arbitrary combination)
    for name, vals in best_mods.iteritems():
        vals['weight'] = (.1 / num_mods) / ((vals['score'] - best_score + .1)*2)
        print vals['weight']
    # create a blended score
    blend_preds = 0
    for name, vals in best_mods.iteritems():
        blend_preds += vals['weight']*vals['preds']
    blend_score = multiclass_log_loss(df_val[target]-1, blend_preds)
    print "Blended score is: " + str(blend_score)
  
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
    for name, val in models.iteritems():
        # isolate test features
        testX = df_test[feats]
        # create weight predictions
        scaled_preds = val['weight']*val['model'].predict_proba(testX)
        # Create weighted average of models in sample sub
        sample_sub.ix[:, 1:] += scaled_preds
    print sample_sub.dtypes
    sample_sub['id'] = sample_sub['id'].astype(int)
    print sample_sub.dtypes
    # export submission
    sample_sub.to_csv(subm_path + nm + '.csv', index=False) 

def blend_subs(files, subm_path, name, targets):
    """ 
    Blends Kaggle submissions and outputs correlations and blended
    NOTE: for now, rates should sum to one
    """
    # create submission template
    keys = list(files.keys())
    blended = pd.read_csv(subm_path+keys[0]+'.csv')
    blended[targets] = 0
    # identify number of targets
    num_trgts = len(targets)
    # for each target and file, blend predictions
    for path, rate in files.iteritems():
        # Created blended value
        subm = pd.read_csv(subm_path+path+'.csv')
        for target in targets:
            blended[target] += rate*subm[target]/num_trgts
    # Save new file back to submissions
    blended.to_csv(subm_path + name + '.csv', index=False) 
    
def subm_correlation(file1, file2, subm_path, targets):
    subm1 = pd.read_csv(subm_path+path+'.csv')
    subm2 = pd.read_csv(subm_path+path+'.csv')
 
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
'boost2' : {'prms' : [400, 3, .10], 'model' : 'none', 'type' : 'boost'},
'boost2a' : {'prms' : [400, 3, .12], 'model' : 'none', 'type' : 'boost'},
'boost4' : {'prms' : [150, 10, .042], 'model' : 'none','type' : 'boost'},
'boost5' : {'prms' : [150, 10, .044], 'model' : 'none', 'type' : 'boost'},
'boost6' : {'prms' : [150, 10, .046], 'model' : 'none', 'type' : 'boost'}
}

mods = {
'svm1' : {'prms' : [9, 'rbf' ], 'model' : 'none', 'type' : 'svm'},
'svm2' : {'prms' : [11, 'rbf'  ], 'model' : 'none', 'type' : 'svm'},
'svm3' : {'prms' : [13, 'rbf'  ], 'model' : 'none', 'type' : 'svm'},
'svm4' : {'prms' : [15, 'rbf' ], 'model' : 'none', 'type' : 'svm'},
'svm5' : {'prms' : [17, 'rbf' ], 'model' : 'none', 'type' : 'svm'}
}

# load data with simple cleaning
df_train = prep_data(PATH+PATH2+"../01 Raw Datasets/", "train.csv")
df_test = prep_data(PATH+PATH2+"../01 Raw Datasets/", "test.csv", is_train=0)
# build variables
df_train = build_Otto_vars(df_train)
df_test = build_Otto_vars(df_test)
Xfeats = select_feats(df_train, "target_num")
# fit models  
fit_models = baseline_models(df_train, Xfeats, mods, target='target_num')
best_mods = evaluate_models(df_train, fit_models, Xfeats, target="target_num")
# blend models
eval_blend_best(df_train, best_mods, target='target_num')
# Create submission
create_subm(best_mods, df_test, Xfeats, PATH + SUBM_PATH, 'sixth svm')

# Blend submission
targets = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5",
           "Class_6", "Class_7", "Class_8", "Class_9"]
subms = {'third svm': .25, 'fourth svm': .25,
         'fifth svm': .25, 'sixth svm': .25}
blend_subs(subms, PATH+SUBM_PATH, 'svm big blended', targets)

        
    
    

