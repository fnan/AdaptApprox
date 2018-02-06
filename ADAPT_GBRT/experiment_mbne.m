function experiment_mbne_em(param_file, param_index, em_round)
% Adaptive Approximation + greedymiser: 

%####################### read parameter file
% lambda, learningrate, p_full, nTrees, max_em_iter, interval, depth
disp('reading parameter file...');

params= dlmread(param_file);
lambda = params(param_index, 1);
learningrate = params(param_index, 2);
p_full = params(param_index, 3);
nTrees = params(param_index, 4);
max_em_iter = params(param_index, 5);
interval = params(param_index, 6);
depth = params(param_index,7);

fprintf('parameters set: %f,%f,%f,%d,%d,%d,%d\n',lambda, learningrate, p_full, nTrees, max_em_iter, interval, depth);
%####################### check warm start file
warm_start_file = sprintf('%s_%d_%d.mat', param_file, param_index, em_round-1);
warm_start = false;
if exist(warm_start_file, 'file') == 2
	load(warm_start_file);
	warm_start = true;
	eall_gate = cell2mat(ensembles_gate{2}');
	eall_clf = cell2mat(ensembles_clf{2}');
	[initpreds_gate, initpreds_clf, ~]= eval_gate_clf_c(xtr,eall_gate, eall_clf,depth,learningrate,cost, interval);
	
	disp('warm start file found and loaded');	
else
	disp('warm start file not found');
end
%####################### read inputs
disp('loading inputs...');
load('mbne_cs_em.mat');
proba_pred_train = dlmread('D:\Feng\C_implement\data\mbne_cs_tr_40_proba_pred_tr');
proba_pred_val = dlmread('D:\Feng\C_implement\data\mbne_cs_tr_40_proba_pred_tv');
proba_pred_test = dlmread('D:\Feng\C_implement\data\mbne_cs_tr_40_proba_pred_te');
feature_usage_val = dlmread('D:\Feng\C_implement\data\mbne_cs_tr_40_featMatrix_tv');
feature_usage_test = dlmread('D:\Feng\C_implement\data\mbne_cs_tr_40_featMatrix_te');
disp('inputs loaded...');


indices = sub2ind(size(proba_pred_train), 1:size(proba_pred_train,1), ytr'*0.5+1.5);
proba_train = proba_pred_train(indices)';
full_pred_train = (proba_pred_train(:,2)>proba_pred_train(:,1))*2-1;
proba_val = proba_pred_val(:,2);
proba_test = proba_pred_test(:,2);
full_pred_val = (proba_pred_val(:,2)>proba_pred_val(:,1))*2-1;
full_pred_test = (proba_pred_test(:,2)>proba_pred_test(:,1))*2-1;


%####################### prepare outputs
output_file = sprintf('%s_%d_%d.mat', param_file, param_index, em_round);
num_settings =1;
totalpreds = nTrees/interval;

ValAccu=zeros(num_settings,totalpreds);
ValCost=zeros(num_settings,totalpreds);
TestAccu=zeros(num_settings,totalpreds);
TestCost=zeros(num_settings,totalpreds);    
ensembles_gate = cell(num_settings,2);
ensembles_clf = cell(num_settings,2);

%####################### start training
assert(min(full_pred_train)==-1,'full_pred_train should have value -1/1');
assert(size(ytr,2) == 1, 'label should be column vector');
assert(size(xtr,1)==size(ytr,1), 'dimension of xtr or ytr is wrong');

disp('start training...');
for iter=1:num_settings 
    options.output_file = output_file;
	options.lambda = lambda;
	options.learningrate=learningrate;	% learning rate
    options.p_full = p_full; % probability of using the full classifier
	options.ntrees = nTrees;		% total number of CART trees
    options.max_em_iter=max_em_iter;
	options.interval = interval;
	options.depth=depth;			% CART tree depth
	if warm_start
		options.initpreds_gate = initpreds_gate;
		options.initpreds_clf = initpreds_clf;
    else
        options.initpreds_gate = zeros(length(ytr),1);
        %initialize using greedymiser
        options.computefeaturecosts = @(e) computefeaturecosts(lambda,cost,e);    
        [~,~,options.initpreds_clf] = gbrt(xtr,@(p)logisticloss(ytr,p),options);
	end
    options.q_threshold = length(ytr)*1e-4;
    options.accu_threshold = 1e-5;
	options.verbose = true;		% verbose on
    
    
	options.computefeaturecosts = @(ensemble_gate, ensemble_clf) computefeaturecosts_gate_clf(lambda,cost,ensemble_gate, ensemble_clf); 	% feature cost update function
	[ensemble_gate, ensemble_clf] = em_adaptive_gbrt(xtr,ytr,proba_train, full_pred_train, options);	
	
    ensembles_gate(iter,:) = ensemble_gate;
    ensembles_clf(iter,:) = ensemble_clf;
    
    [tst_accu,tst_cost, tst_p_full, val_accu, val_cost, val_p_full] = eval_gate_clf_BC(ensemble_gate,ensemble_clf, options,cost,xtv,ytv,xte,yte, full_pred_val, full_pred_test, feature_usage_val, feature_usage_test);
    ValAccu(iter,:)=val_accu;
    ValCost(iter,:)=val_cost;
    TestAccu(iter,:)=tst_accu;
    TestCost(iter,:)=tst_cost;
    disp([param_file, ', param index=', num2str(param_index), ', em_round=', num2str(em_round), ', lambda= ',num2str(lambda),', p_full=', num2str(p_full), ', learningrate=',num2str(learningrate)]);
    disp(['  val accu: ', num2str(val_accu)]);
    disp(['  val cost: ', num2str(val_cost)]);
    disp(['  tst accu: ', num2str(tst_accu)]);
    disp(['  tst cost: ', num2str(tst_cost)]);
end
save(output_file, 'ensembles_gate', 'ensembles_clf', 'ValAccu', 'ValCost', 'TestAccu', 'TestCost');
disp([output_file, ' saved. Done!']);


