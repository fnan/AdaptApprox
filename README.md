The code for "Adaptive Classification for Prediction Under a Budget", NIPS 2017 by Feng Nan and Venkatesh Saligrama.
Bibtex:
```
@incollection{NIPS2017_7058,
title = {Adaptive Classification for Prediction Under a Budget},
author = {Nan, Feng and Saligrama, Venkatesh},
booktitle = {Advances in Neural Information Processing Systems 30},
editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
pages = {4730--4740},
year = {2017},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7058-adaptive-classification-for-prediction-under-a-budget.pdf}
}
```

---
# ADAPT-LIN:
Contains python codes for adaptive approximation using linear gating and Low-Prediction-Cost (LPC) model. 
`experiment_letters_linear.py` provides an example for binary classification of the [letters dataset](https://archive.ics.uci.edu/ml/datasets/letter+recognition). 
Simply specify the data file location as well as Liblinear library location and run it. 

# ADAPT-GBRT:
Contains Matlab codes for adaptive approximation using gradient boosted trees as gating and LPC model. 

## (Before you can run the Matlab code) MEX files:
The following sub-routines are written in c code to speed up computation. Please use mex utility in MATLAB to compile them into excutables e.g. .mexa64 or .mexw64.
- `eval_gate_clf_c.c`
- `buildlayer_sqrimpurity_openmp.cpp`

`experiment_mbne.m` provides an example for binary classification of the [MiniBooNE dataset](https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification)
There are 3 inputs to run this program and several others to be supplied in a `.mat` file:

## INPUTS:
1. `param_file`: each row corresponds to a setting in the order of (Lambda, LearningRate, P_full, nTrees, max_em_iter, interval, depth). A sample parameter generating file is `mbne_param_gen.m`.
..* `Lambda`: multiplier of the feature acquisition costs
..* `LearningRate`: learning rate for the gradient boosted trees
..* `P_full`: fraction of examples to be sent to the complex classifier
..* `nTrees`: number of trees for gating function, same number of trees for low prediction cost model
..* `max_em_iter`: maximun number of alternating minimization iterations
..* `interval`: evaluation interval in the number of trees. The outputs will be evaluated using the first interval, 2*interval, ... trees for gating and LPC. Total number of evaluations will be nTrees/interval.
..* `depth`: depth of trees for gating and LPC

2. `setting_id`: the row number in param_file to execute

3. last parameter is for warm start, it can be set as a constant 1

Other required inputs: (specified in line 34-39 of experiment_mbne.m)

4. data_file: mat file containing the basic input data to the algorithm. See mbne_cs_em.mat for example.
..* xtr: training data, dimension = # training examples x # features
..* xtv: validation data, dimension = # validation examples x # features
..* xte: test data, dimension = # test examples x # features
..* ytr: class label. -1/1 for binary classification, dimension = # training examples x 1
..* ytv: class label. -1/1 for binary classification, dimension = # validation examples x 1
..* yte: class label. -1/1 for binary classification, dimension = # test examples x 1
..* cost: feature acquisition cost vector, dimension = # features x 1
..* proba_pred_train: probability of class prediction from the High-Prediction-Cost (HPC) model, dimension = # training examples x # classes
..* proba_pred_val: probability of class prediction from the High-Prediction-Cost (HPC) model, dimension = # validation examples x # classes
..* proba_pred_test: probability of class prediction from the High-Prediction-Cost (HPC) model, dimension = # test examples x # classes
..* feature_usage_val: feature usage matrix for validation data by HPC model, dimension = # validation examples x # features. (i,j) element is 1 if feature j is used for example i by HPC; otherwise it is 0.
..* feature_usage_test: feature usage matrix for test data by HPC model, dimension = # test examples x # features. (i,j) element is 1 if feature j is used for example i by HPC; otherwise it is 0.

## OUTPUTS:
results are saved into file that contains:
1. ensembles_gate: the learned gating ensemble
2. ensembles_clf: the learned LPC ensemble
3. ValAccu: the accuracy on validation data
4. ValCost: the feature acquisition costs on validation data
5. TestAccu: the accuracy on test data
6. TestCost: the feature acquisition costs on test data



