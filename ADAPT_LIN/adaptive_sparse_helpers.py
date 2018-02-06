# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 15:09:48 2017

@author: fnan
"""

from sklearn.linear_model import LogisticRegression, LinearRegression
import collections
import cPickle, gzip
import numpy as np
import scipy.sparse
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import math
from sklearn.ensemble import RandomForestClassifier
import scipy.io as sio
import sys
sys.path.insert(0, 'liblinear-multicore-2.11-1\python')
from liblinearutil import *
#import matlab.engine
import multiprocessing
from functools import partial
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

def ismember(A, B):
    B = np.array(B)
    return [ np.sum(a == B) for a in A ]
            
			
def load_letter():
    with open('letter-recognition.data', 'r') as f:
        M = [ line.split(',') for line in f ]
    letters = np.array([x[0]>= 'N' for x in M])
    features = [x[1:] for x in M]
    M = np.array(features).astype(float)
    ntr = 12000
    X_train = M[:ntr,:]
    X_val = M[ntr:16000,:]
    X_test = M[16000:,:]
    y_train = (letters[:ntr]+0).astype(int)
    y_val = (letters[ntr:16000]).astype(int)
    y_test = (letters[16000:]+0).astype(int)
    cost = np.ones((M.shape[1],1))

    X_max = np.amax(abs(X_train),axis=0)
    X_train = X_train/X_max
    X_test = X_test/X_max
    X_val = X_val/X_max
           
    return X_train, X_test, X_val, y_train, y_test, y_val, cost

	    
def em_adaptive_sparse(lambd, p_full, X_train, y_train,proba_train, h_init):
    assert (np.amin(y_train) == -1 and np.amax(y_train) == 1), "Label has to be -1/1 for training"
    #########################################
    #########    Perform alternating minimization to learn g,h ####
    maxIter=50
    ntr, m = X_train.shape
    #gating function initialization
    #g = np.random.normal(0,1,m)
    g = np.zeros(m)
    g0 = 0
    
    #partial classifier initialization
    h_copy_init = h_init.copy()
    h = h_copy_init[:-1]
    h0 = h_copy_init[-1]
    
    for em_iter in range(maxIter):
        ######## E-step
        pz1 = 1/(1+np.exp(-X_train.dot(g)-g0))
        pz0 = 1 - pz1
        
        pyz1 = proba_train
#        pyz1 = np.zeros(len(proba_train))
#        msk = proba_train>0.5
#        pyz1[msk] = proba_train[msk]
        pyz0 = 1/(1+np.exp(-y_train*(X_train.dot(h)+h0)))
        
        w1 = pyz1*pz1
        w0 = pyz0*pz0
        
        # I-projection to satisfy the posterior constraint
        # use binary search for gamma
        g_max_cur = 1e5
        g_min_cur = 1e-5
        gamma_threashold = 1e-3
        max_gamma_iter = 100
        
        for gamma_iter in range(max_gamma_iter):
            gamma = 0.5*(g_max_cur+g_min_cur)
            frac_cur = np.sum(w1*gamma/(w1*gamma+w0))
            if frac_cur > p_full*ntr:
                g_max_cur = gamma
            else:
                g_min_cur = gamma
            if np.abs(frac_cur-p_full*ntr) < gamma_threashold:
                break
        
        Z = w1*gamma + w0
        q1 = w1*gamma/Z
        q0 = w0/Z
#        print "gamma = %e, I-projection error: q1.sum()/ntr-p_full = %f" %(gamma, q1.sum()/ntr-p_full)
        
        ######## M-step
        gh = np.concatenate((g,np.array([g0,]),h,np.array([h0,])))
        obj_bound = partial(em_adaptive_sparse_obj,X_train, y_train, q0,q1, lambd)
        der_bound = partial(em_adaptive_sparse_der,X_train, y_train, q0,q1, lambd)
        
        res = minimize(obj_bound, gh, method='BFGS', jac=der_bound, options={'disp': False})
#        print "EM iter:", em_iter
        gh = res.x        
#        print gh
        len_gh = len(gh)
        g = gh[:len_gh/2-1]
        g0 = gh[len_gh/2-1]
        h = gh[len_gh/2:-1]
        h0 = gh[-1]
    return gh

def l1_adaptive_sparse_eval(gh_in,X,y,full_pred): #sweeps across the support sizes
    gh = gh_in.copy()
    len_gh = len(gh)
    g = gh[:len_gh/2-1]
    g0 = gh[len_gh/2-1]
    h = gh[len_gh/2:-1]
    h0 = gh[-1]

    cost_full = len(g)    
    ground_truth = (y==1)+0
    inactive_feature_mask = (abs(g)+abs(h))<1e-8
    cost_partial = sum(~inactive_feature_mask)
    gating_output = (X.dot(g)+g0 > 0)+0
    partial_clf_output =(X.dot(h)+h0 > 0)+0
    final_pred = full_pred.copy()
    final_pred[gating_output==0]=partial_clf_output[gating_output==0]
    frac_to_partial = sum(gating_output==0)*1.0/len(y)
    final_accu = sum(final_pred==ground_truth)*1.0/len(y)
    final_cost = frac_to_partial*cost_partial + (1-frac_to_partial)*cost_full
        
    return 1-frac_to_partial, final_accu , final_cost
    
def em_adaptive_sparse_eval(gh_in,X,y,full_pred): #sweeps across the support sizes
    gh = gh_in.copy()
    len_gh = len(gh)
    g = gh[:len_gh/2-1]
    g0 = gh[len_gh/2-1]
    h = gh[len_gh/2:-1]
    h0 = gh[-1]
    mag_v = np.sqrt(g*g+h*h)
    sorted_mag_v = np.sort(mag_v)
    
    frac_to_partial = np.zeros(len(g))
    final_accu = np.zeros(len(g))
    final_cost = np.zeros(len(g))

    cost_full = len(g)    
    ground_truth = (y==1)+0
    i = 0
    for support_cutoff_threashold in sorted_mag_v:
        inactive_feature_mask = mag_v <= support_cutoff_threashold
        g[inactive_feature_mask]=0
        h[inactive_feature_mask]=0
        cost_partial = sum(~inactive_feature_mask)
        gating_output = (X.dot(g)+g0 > 0)+0
        partial_clf_output =(X.dot(h)+h0 > 0)+0
        final_pred = full_pred.copy()
        final_pred[gating_output==0]=partial_clf_output[gating_output==0]
        frac_to_partial[i] = sum(gating_output==0)*1.0/len(y)
        final_accu[i] = sum(final_pred==ground_truth)*1.0/len(y)
        final_cost[i] = frac_to_partial[i]*cost_partial + (1-frac_to_partial[i])*cost_full
        i = i+1
        
    return 1-frac_to_partial, final_accu , final_cost

def em_adaptive_sparse_obj(X_train, y_train, q0,q1, lambd, gh):
    len_gh = len(gh)
    g = gh[:len_gh/2-1]
    g0 = gh[len_gh/2-1]
    h = gh[len_gh/2:-1]
    h0 = gh[-1]
    sum1 = np.sum(q1*np.log(1+np.exp(-X_train.dot(g)-g0))+q0*(np.log(1+np.exp(-y_train*(X_train.dot(h)+h0)))+np.log(1+np.exp(X_train.dot(g)+g0))))
    sum2 = np.sum(np.sqrt(g*g+h*h))
    return sum1/len(y_train)+lambd*sum2

def em_adaptive_sparse_der(X_train, y_train, q0,q1, lambd, gh):
    len_gh = len(gh)
    g = gh[:len_gh/2-1]
    g0 = gh[len_gh/2-1]
    h = gh[len_gh/2:-1]
    h0 = gh[-1]
    v1 = - q1 * 1 / (1+np.exp(X_train.dot(g)+g0)) + q0 * 1 / (1+np.exp(-X_train.dot(g)-g0))    
    v2 = np.sqrt(g*g + h*h)
    g_der = v1.dot(X_train)/len(y_train) + np.nan_to_num(lambd* g / v2)

    g0_der = np.sum(v1)/len(y_train)

    v3 = - y_train * q0 * 1 / (1+np.exp(y_train*(X_train.dot(h)+h0))) 
    h_der = v3.dot(X_train)/len(y_train) + np.nan_to_num(lambd* h / v2)
    h0_der = np.sum(v3)
    return np.concatenate((g_der, np.array([g0_der,]), h_der, np.array([h0_der,])))

def get_full_rf_clf(dataset, X_train, y_train, retrain=False):
    if retrain:
        clf = RandomForestClassifier(n_estimators=1000,criterion="entropy",n_jobs=-1)
        clf.fit(X_train, y_train)
#        full_predict_test = clf.predict(X_test)
#        print sum(y_test==full_predict_test)*1.0/len(y_test)        
        joblib.dump(clf, '%s_rf1000.pkl'%dataset)
    else:
        clf = joblib.load('%s_rf1000.pkl'%dataset)
        
    return clf
#        clf_probs = clf.predict_proba(X_test)
    
def get_full_rbf_svm_clf(dataset, X_train, y_train, retrain=False, C_range=None, gamma_range=None):
    if retrain:
#        if C_range is None:
#            C_range = np.logspace(-2, 10, 13)
#        if gamma_range is None:
#            gamma_range = np.logspace(-9, 3, 13)
        if C_range is None:
            c_best = 1
            gamma_best = 1
        else:
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(SVC(verbose=True), param_grid=param_grid, cv=cv, n_jobs=40,verbose=10)
            grid.fit(X_train, y_train)
            
            print("The best parameters are %s with a score of %0.2f"
                  % (grid.best_params_, grid.best_score_))
            c_best = grid.best_params_['C']
            gamma_best = grid.best_params_['gamma']

        clf = SVC(C=c_best, gamma=gamma_best, probability=True)
        clf.fit(X_train, y_train)    
        joblib.dump(clf, '%s_rbf_svm.pkl'%dataset)
        
        return clf
    else:
        clf = joblib.load('%s_rbf_svm.pkl'%dataset)
        return clf

def get_l1_gate_partial_clf(dataset, X_train, y_train, c_array, retrain=False):
    if retrain:
        l1_dic = l1_adaptive_sparse(X_train, y_train,c_array)
        with open(dataset+"_gh_dic_l1.pickle", "wb") as output_file:
            cPickle.dump(l1_dic, output_file)
    else:
        with open(dataset+"_gh_dic_l1.pickle", "rb") as input_file:
            l1_dic = cPickle.load(input_file)
    return l1_dic
    
def get_em_gate_partial_clf(dataset, X_train, y_train, proba_train, lambd_p_array, h_init, retrain=False):
    if retrain:
        output_dic = {}
        print "Training EM for %s, %d lambd_p settings"%(dataset, len(lambd_p_array))
        counter=0
        for lambd_p in lambd_p_array:
            lambd = lambd_p[0]
            p_full = lambd_p[1]
            gh = em_adaptive_sparse(lambd, p_full, X_train, y_train,proba_train,h_init)    
            output_dic[lambd_p] = gh
            print "%d out of %d done"%(counter, len(lambd_p_array))
            counter = counter+1
        with open(dataset+str(len(lambd_p_array))+"_em_rf.pickle", "wb") as output_file:
            cPickle.dump(output_dic, output_file)
        return output_dic
    else:
        with open(dataset+str(len(lambd_p_array))+"_em_rf.pickle", "rb") as input_file:
            output_dic = cPickle.load(input_file)
        return output_dic
        
def l1_adaptive_sparse(X_train, y_train, c_array):
#    assert (np.amin(y_train) == 0 and np.amax(y_train) == 1), "Label has to be 0/1 for training"
    m = X_train.shape[1]
    l1_dic={}
    learn_partial_gating_bound = partial(learn_partial_gating_dec_funcs, X_train, y_train)
    counter= 0
    for c in c_array:
        liblinear_param_l1 = parameter('-s 6 -c %f -B 1 -q'%c)
        prob_l1 = problem(y_train, X_train.tolist())
        liblinear_model_l1 = train(prob_l1, liblinear_param_l1)
        decfun_l1 = liblinear_model_l1.get_decfun()[0]
        sortInd_l1 = np.argsort(abs(np.array(decfun_l1)))
        for top_n_features in range(m-1):
            top_n_features = top_n_features+1
            support_l1 = sortInd_l1.copy()[-top_n_features:]
            support_l1 = ismember(range(m), support_l1)
            support_l1 = np.array(support_l1)>0
            if support_l1.tostring() not in l1_dic:
                gh_array = learn_partial_gating_bound(support_l1)
                l1_dic[support_l1.tostring()] = gh_array
        counter = counter +1
        print "%d out of %d are done"%(counter, len(c_array))
                   
    return l1_dic
    
def learn_partial_gating_dec_funcs(X_train, y_train, support):
    ################### learn partial linear classifier ###################
#    liblinear_param = parameter('-s 0 -c 1e-6 -B 1 -C -q')
    prob = problem(y_train.tolist(), X_train[:,support].tolist())
#    best_param = train(prob, liblinear_param)
#    best_c = best_param[0]
    best_c = 1e0
    liblinear_param = parameter("-s 0 -c %f -B 1 -q" % best_c)
    liblinear_model = train(prob, liblinear_param)
    y_train_pred_partial,p_acc, p_train = predict(y_train.tolist(), X_train[:, support].tolist(), liblinear_model)
    yhat_train_sparse = np.array(y_train_pred_partial)
    h_dec_func = liblinear_model.get_decfun()
        
    m = X_train.shape[1]
    h = np.zeros(m)    
    if liblinear_model.label.contents.value == 1:
        h[support] = np.array(h_dec_func[0])
        h0 = h_dec_func[1]
    else:
        h[support] =  - np.array(h_dec_func[0])
        h0 = - h_dec_func[1]

    ################## Learn gating linear classifiers ####################
    n_tradeoff_points=10
    gh_array = np.zeros((n_tradeoff_points, len(h)*2+2))
    
    lambd_array = np.linspace(0,1,num=n_tradeoff_points)**2
    pseudo_labels=(-1)**(yhat_train_sparse==y_train)
    i =0
    for lambd in lambd_array:
#        liblinear_param_gate = parameter('-s 0 -c 1e-6 -B 1 -w0 %f -w1 1 -n 20 -q -C' %lambd )
        prob_gate = problem(pseudo_labels.tolist(), X_train[:,support].tolist())
#        best_param_gate = train(prob_gate, liblinear_param_gate)
#        best_c_gate = best_param_gate[0]
        best_c_gate = 1e0
        liblinear_param_gate = parameter('-s 0 -c %f -B 1 -w-1 %f -w1 1 -n 20 -q' %(best_c_gate,lambd))        
        liblinear_model_gate = train(prob_gate, liblinear_param_gate)
        
        g_dec_func = liblinear_model_gate.get_decfun()
        g = np.zeros(m)
        if liblinear_model_gate.label.contents.value == 1:
            g[support] = np.array(g_dec_func[0])
            g0 = g_dec_func[1]
        else:
            g[support] = - np.array(g_dec_func[0])
            g0 = - g_dec_func[1]
            
        gh_array[i,:] = np.concatenate((g,np.array([g0,]),h,np.array([h0,])))
        i = i+1

    return gh_array

def l1_adaptive_sparse_eval_all(dataset, l1_dic, X, y, full_predict, load_existing=True, hull_indices=None):    
    if load_existing:
        val_eval_array = np.load(dataset+"_%d_eval_l1"%len(y))
    else:
        key_list = l1_dic.keys()
        sparsity_num = len(l1_dic[key_list[0]])
        val_eval_array= np.zeros((len(key_list)*sparsity_num,3))
        key_i =0 
        for support_key in key_list:
            gh_array = l1_dic[support_key]
            gh_i = 0
            for gh in gh_array:
                if hull_indices is None:
                    frac_to_full, final_accu, final_cost = l1_adaptive_sparse_eval(gh,X,y,full_predict)
                    val_eval_array[key_i*sparsity_num+gh_i,:] = np.array([frac_to_full, final_accu, final_cost])                
                if hull_indices is not None and key_i*sparsity_num+gh_i in hull_indices:
                    frac_to_full, final_accu, final_cost = l1_adaptive_sparse_eval(gh,X,y,full_predict)
                    val_eval_array[key_i*sparsity_num+gh_i,:] = np.array([frac_to_full, final_accu, final_cost])                
                gh_i = gh_i+1
            key_i = key_i + 1
        np.save(dataset+"_%d_eval_l1"%len(y), val_eval_array)
            
    pts = val_eval_array[:,(1,2)]
    return pts
    
def em_adaptive_sparse_eval_all(dataset, output_dic, X, y, full_predict, lambd_p_array, load_existing=True, hull_indices=None):
    if load_existing:
        val_eval_array = np.load(dataset+"_%d_eval_em"%len(y))
    else:            
        key_list = output_dic.keys()
        m = X.shape[1]
        val_eval_array= np.zeros((len(key_list)*m,3))
        lambd_p_hull=np.zeros(0)
        if hull_indices is not None:
            lambd_p_hull = hull_indices/m
        lambd_p_i = 0
        for lambd_p in key_list:
            if hull_indices is None:
                gh = output_dic[lambd_p]
                frac_to_full, final_accu, final_cost = em_adaptive_sparse_eval(gh,X,y,full_predict)
                val_eval_array[lambd_p_i*m:(lambd_p_i+1)*m,:] = np.vstack((frac_to_full, final_accu, final_cost)).transpose()
                
            if hull_indices is not None and lambd_p_i in lambd_p_hull:
                gh = output_dic[lambd_p]
                frac_to_full, final_accu, final_cost = em_adaptive_sparse_eval(gh,X,y,full_predict)
                val_eval_array[lambd_p_i*m:(lambd_p_i+1)*m,:] = np.vstack((frac_to_full, final_accu, final_cost)).transpose()
            lambd_p_i = lambd_p_i +1
        np.save(dataset+"_%d_eval_em_lin_rf"%len(y), val_eval_array)
        
    pts = val_eval_array[:,(1,2)]
    return pts

def get_efficient_frontier_accu_cost(pts):
    # plot the convex hull of all validation results:
    ch = ConvexHull(pts)
    
    # Get the indices of the hull points.
    hull_indices = ch.vertices
    
    # These are the actual points.
    hull_pts = pts[hull_indices, :]
    
    # remove the lower points, only get the efficient frontier 
    anything_removed = True
    while anything_removed:
        anything_removed = False
        for pt_indx in range(len(hull_indices)):
            for cmp_indx in range(len(hull_indices)):
                if hull_pts[pt_indx,0] < hull_pts[cmp_indx, 0] and hull_pts[pt_indx,1] > hull_pts[cmp_indx,1]:
                    hull_indices = np.delete(hull_indices, pt_indx)
                    hull_pts = np.delete(hull_pts, (pt_indx), axis=0)
                    anything_removed = True
                    break
            if anything_removed:
                break

    return hull_indices