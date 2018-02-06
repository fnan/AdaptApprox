function [tst_accu,tst_cost, tst_p_full, val_accu, val_cost, val_p_full] = eval_gate_clf_BC(ensemble_gate,ensemble_clf, options,cost,xtv,ytv,xte,yte, val_full_pred, tst_full_pred,feature_usage_val, feature_usage_test)
	% evaluation code, we evaluate at every 10 trees
	
	% input:
	% e, greedy miser trees
	% options, greedy miser tree information
	% cost = d*1, cost information
	% xtr = n*d, ytr = n*1, 
	% traqs = n*1, queries only valid for ranking data.
	
	% output:
	% tst_prec: testing accuracy at every 10 trees, n*(ntrees/10)
	% val_prec: validation accuracy at every 10 trees, n*(ntrees/10)
	% beststep: best number of trees based on validation set
	% totalcost: total cost at every 10 trees, 1*(ntrees/10);
	assert(min(tst_full_pred)==-1 && min(val_full_pred)==-1, 'tst and val full_pred must be -1/1');
	eall_gate = cell2mat(ensemble_gate{2}');
	eall_clf = cell2mat(ensemble_clf{2}');
	% we evaluate at every 10 trees
%	[tst_preds, totalcost]= evalensemble_feature_c(xte',eall,options.depth,options.learningrate,10,cost);
	[tst_preds_gate, tst_pred_clf, tst_cost_m]= eval_gate_clf_c(xte,eall_gate, eall_clf,options.depth,options.learningrate,cost, options.interval);

	[val_preds_gate, val_pred_clf, val_cost_m]= eval_gate_clf_c(xtv,eall_gate, eall_clf,options.depth,options.learningrate,cost, options.interval);
    totalpreds = 1;
    if options.interval~=0
        totalpreds = options.ntrees/options.interval;
    end
    tst_preds_gate = cumsum(tst_preds_gate,2);
    tst_pred_clf = cumsum(tst_pred_clf,2);
    tst_cost_m = cumsum(tst_cost_m,3);
    tst_cost_m = (tst_cost_m >0 )+0;
    
    val_preds_gate = cumsum(val_preds_gate,2);
    val_pred_clf = cumsum(val_pred_clf,2);
    val_cost_m = cumsum(val_cost_m,3);
    val_cost_m = (val_cost_m >0)+0;
    
    % to be returned:
    tst_accu = zeros(1, totalpreds);
    tst_cost = zeros(1, totalpreds);
    tst_p_full =zeros(1, totalpreds);
    val_accu =zeros(1, totalpreds);
    val_cost =zeros(1, totalpreds);
    val_p_full =zeros(1, totalpreds);
    
    feature_usage_test = (feature_usage_test>0)+0;
    feature_usage_val= (feature_usage_val>0)+0;
    for i = 1:totalpreds
        tst_gate_tmp =tst_preds_gate(:,i)>0;
        tst_p_full(i) = sum(tst_gate_tmp)/length(yte);
        tst_pred_tmp = (tst_pred_clf(:,i)>0)*2-1;
        tst_pred_tmp(tst_gate_tmp) = tst_full_pred(tst_gate_tmp);
        tst_cost_m(tst_gate_tmp,:,i) = feature_usage_test(tst_gate_tmp,:);    
        tst_cost(i) = mean(tst_cost_m(:,:,i)*cost);    
        tst_accu(i) = sum(yte==tst_pred_tmp)/length(yte);
    
        val_gate_tmp =val_preds_gate(:,i)>0;
        val_p_full(i) = sum(val_gate_tmp)/length(ytv);
        val_pred_tmp = (val_pred_clf(:,i)>0)*2-1;
        val_pred_tmp(val_gate_tmp) = val_full_pred(val_gate_tmp);   
        val_cost_m(val_gate_tmp,:,i)=feature_usage_val(val_gate_tmp,:);
        val_cost(i) = mean(val_cost_m(:,:,i)*cost);    
        val_accu(i) = sum(ytv==val_pred_tmp)/length(ytv);
    end
end