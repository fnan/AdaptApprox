# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 17:44:09 2017

@author: fnan
"""
# load dataset
import numpy as np
from adaptive_sparse_helpers import load_letter, em_adaptive_sparse_obj, em_adaptive_sparse_der, em_adaptive_sparse_eval,em_adaptive_sparse,get_efficient_frontier_accu_cost,em_adaptive_sparse_eval_all,get_em_gate_partial_clf, get_full_rbf_svm_clf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from functools import partial
import matplotlib.pyplot as plt
import cPickle
from sklearn.externals import joblib
import sys
sys.path.insert(0, 'liblinear-multicore-2.11-1\python')
from liblinearutil import *

if __name__ == '__main__':
    
    ################### load data ###################
    [X_train, X_test, X_val, y_train, y_test, y_val, cost] = load_letter()
    ntr, m = X_train.shape
    assert (type(X_train) is np.ndarray and type(y_train) is np.ndarray and type(X_test) is np.ndarray and type(y_test) is np.ndarray ), "All data inputs need to be numpy.ndarray"
    assert (type(y_train[0]) is np.int32 and type(y_test[0]) is np.int32), "Label has to be of type numpy.int32"
    assert (np.amin(y_train) == 0 and np.amin(y_test) == 0 and np.amax(y_train) == 1 and np.amax(y_test) ==1 ), "Label has to be 0/1"
    assert (m == X_test.shape[1] and ntr == len(y_train) and X_test.shape[0] == len(y_test)), "Input dimension mis-match! Samples should be stored in rows and len(y_train) should match the number of rows of X_train"

    dataset="letters"
    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    #########################################
    ##### Get the full classifier rbf svm ###
    retrain = False # load pre-trained model
    clf = get_full_rbf_svm_clf("letters", X_train, y_train, retrain) 

    proba_train = clf.predict_proba(X_train)[np.arange(ntr),(y_train==1)+0]
    #########################################
    #########    Perform alternating minimization to learn g,h ####
    y_train=(y_train-0.5)*2 # make y_train -1/1

    lambd_array = np.logspace(-4,0,num=20) #group sparsity parameter
    p_full_array = np.linspace(0.1,0.9,9) #fraction of examples to go to full classifier
    lambd_p_array = [(x,y) for x in lambd_array for y in p_full_array]
    output_dic={}

    prob = problem(y_train.tolist(), X_train.tolist())
    best_c = 1e0
    liblinear_param = parameter("-s 0 -c %f -B 1 -q" % best_c)
    liblinear_model = train(prob, liblinear_param)
    
    h_init0 = np.array(liblinear_model.get_decfun()[0])
    h_init1 = liblinear_model.get_decfun()[1]
    h_init = np.concatenate((h_init0, np.array([h_init1,])))
    if liblinear_model.label.contents.value != 1:
        h_init = - h_init
        
    retrain = True # load pre-trained model
    output_dic = get_em_gate_partial_clf(dataset, X_train, y_train, proba_train, lambd_p_array, h_init, retrain)
    
        
    ############################################################
    #########    Evaluate all g,h models on validation data ####
    load_existing = False
    pts_val = em_adaptive_sparse_eval_all(dataset, output_dic, X_val, y_val, clf, lambd_p_array, load_existing) 
    
    ####################################################
    #########    Get the indices of the best points ####
    hull_indices = get_efficient_frontier_accu_cost(pts_val)    
    if False:
        keys = output_dic.keys()
        gh_hull = np.zeros((len(hull_indices), len(output_dic[keys[0]])))
        sparsity_num = np.zeros(len(hull_indices))
        i = 0
        fig_dic={}
        axes_dic={}
            
        for hull_index in hull_indices:
            key_num = hull_index / 16
            sparsity_num[i] = hull_index % 16
            gh_in = output_dic[keys[key_num]]
            gh = gh_in.copy()
            len_gh = len(gh)
            g = gh[:len_gh/2-1]
            g0 = gh[len_gh/2-1]
            h = gh[len_gh/2:-1]
            h0 = gh[-1]
            mag_v = np.sqrt(g*g+h*h)
            sorted_mag_v = np.sort(mag_v)
            inactive_feature_mask = mag_v <= sorted_mag_v[sparsity_num[i]]
            g[inactive_feature_mask]=0
            h[inactive_feature_mask]=0
            gh_hull[i,:] = np.concatenate((g, np.array([g0,]), h, np.array([h0,])))
            i = i+1
            fig_dic[hull_index] = plt.figure()
            fig_dic[hull_index].suptitle(str(hull_index)+": accu cost:"+str(pts_val[hull_index,:]), fontsize=20)
            axes_dic[hull_index] = fig_dic[hull_index].add_subplot(111)
            axes_dic[hull_index].plot(abs(g),label='|g|')
            axes_dic[hull_index].plot(abs(h),label='|h|')
            axes_dic[hull_index].legend()
            fig_dic[hull_index].savefig('EM_gh_hull_plot%d.svg' % hull_index, transparent=True, bbox_inches='tight', pad_inches=0)
        
    ############################################################
    #########    Evaluate all g,h models on test data ####
    load_existing = False
    pts_test = em_adaptive_sparse_eval_all(dataset, output_dic, X_test, y_test, clf, lambd_p_array, load_existing, hull_indices) 
    accu_test_hull = pts_test[hull_indices,0]
    cost_test_hull = pts_test[hull_indices,1]
    test_accu_cost={'accu_test_hull':accu_test_hull, 'cost_test_hull':cost_test_hull}
    with open(dataset+"_test_accu_cost_em.pickle", "wb") as output_file:
        cPickle.dump(test_accu_cost, output_file)
    
    with open(dataset+"_test_accu_cost_l1.pickle", "rb") as input_file:
        test_accu_cost_l1 = cPickle.load(input_file)
        
    plt.plot(cost_test_hull,accu_test_hull,"k-", label='EM')
    plt.plot(test_accu_cost_l1['cost_test_hull'], test_accu_cost_l1['accu_test_hull'], "r--",label='L1')      
#    plt.xlim(0,16)
#    plt.ylim(0.5,1)
    plt.ylabel('accuracy')
    plt.xlabel('feature cost')
    plt.title('%s EM' % (dataset))
    plt.legend(loc="lower right")
    plt.savefig('%s_EM_zoom_out_plot.svg' % dataset, transparent=True, bbox_inches='tight', pad_inches=0)
