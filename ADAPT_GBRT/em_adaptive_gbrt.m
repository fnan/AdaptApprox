%% Builds two collections of limited-depth regression trees as gating and partial classifiers as part of EM iterations
%% Returns two cell arrays of trees and the loss value after training.

function [ensemble_gate, ensemble_clf,loss,p_gate, p_clf] = em_adaptive_gbrt(X,y,proba_train, full_pred_train, options)
	% check for required arguments and outputs
	if nargin < 5,
		error('Too few arguments');
	elseif nargin > 5,
		error('Too many arguments');
	end
	
	assert(isa(options,'struct'),'Argument options must be a struct');
	
	% set defaults for options.depth
	if isfield(options,'depth'),
		assert(options.depth >= 0, 'options.depth must be >= 0');
	else
		options.depth = 4;
	end
	
	% set defaults for options.ntrees
	if isfield(options,'ntrees'),
		assert(options.ntrees >= 1, 'options.ntrees must be >= 1');
	else
		options.ntrees = 100;
	end
	
	% set defaults for options.learningrate
	if isfield(options,'learningrate'),
		if ~isa(options.learningrate,'function_handle')
			options.learningrate = @(t) options.learningrate; % constant learning rate
		end
	else
		options.learningrate = @(t) 0.1;
	end
	
	% set default for options.verbose
	if not(isfield(options,'verbose')),
		options.verbose = false;
	end
	
	% sort the training input feature-wise (column-wise)
	[N,nf] = size(X);
	[Xs,Xi] = sort(X);
	
	
	% initialize predictions
    if isfield(options,'initpreds_clf')
		%assert(all(size(options.initpreds) == [N,1]),...
		%	'options.initpreds must be a vector of length size(X,1)');
		p_gate = options.initpreds_gate;
        p_clf = options.initpreds_clf;
    else
		p_gate = zeros(N,1);
        p_clf = zeros(N,1);
    end
    
    q1_old = zeros(size(proba_train));
    accu_old = 0;
	max_em_iter = options.max_em_iter;
    for em_iter = 1:max_em_iter
        %%%% E-step
        [q1, q0] = updateQ(p_gate, p_clf, options.p_full, y,proba_train,N);
        %%%% M-step
        % initialize ensemble (cell array of trees)
        ensemble_gate = {[],{}};
        ensemble_clf = {[],{}};
        % compute initial loss and gradient
        
        % initialize predictions
        p_gate = zeros(N,1);
        p_clf = zeros(N,1);
        
        loss = [];
        [loss(1),gate_der, clf_der] = lossNgrad_gate_clf(y, p_gate, p_clf, q0, q1); 
        
        %if options.verbose, fprintf('Initial loss %f\n',loss(end)); end;

        % construct gating and partial classifier trees to minimize loss
        for t=1:options.ntrees
            % compute feature costs
            options.featurecosts = options.computefeaturecosts(ensemble_gate, ensemble_clf);
            if size(gate_der,2) == 1
                % construct one tree for gating:
                [tree_gate,pt_gate] = buildtree(X,Xs,Xi,gate_der,options.depth,options);
                p_gate(:) = p_gate(:) + options.learningrate(t) * pt_gate(:); % update predictions for gate
                ensemble_gate{1}(t) = options.learningrate(t); % add learning rate to ensemble
                ensemble_gate{2}{t} = tree_gate; % add tree to ensemble
                
                options.featurecosts = options.computefeaturecosts(ensemble_gate, ensemble_clf);
                % construct one tree for partial classifier:
                [tree_clf,pt_clf] = buildtree(X,Xs,Xi,clf_der,options.depth,options);                
                p_clf(:) = p_clf(:) + options.learningrate(t) * pt_clf(:); % update predictions for partial classifier
                ensemble_clf{1}(t) = options.learningrate(t); % add learning rate to ensemble
                ensemble_clf{2}{t} = tree_clf; % add tree to ensemble
            else
                error('!Gradient has to be a vector. Multi-class is currently not supported!');
            end
            %update q1 and q0
%            [q1, q0] = updateQ(p_gate, p_clf, options.p_full, y,proba_train,N);
            % update predictions and ensemble
            %[q1, q0] = updateQ(p_gate, p_clf, options.p_full, y,proba_train,N);
            [loss(t+1),gate_der, clf_der] = lossNgrad_gate_clf(y, p_gate, p_clf, q0, q1); 
%            if options.verbose, fprintf('After tree %d, loss %f\n', t, loss(end)); end;
        end        
        pred_final = full_pred_train;
        to_clf = p_gate<0;
        pred_clf = (p_clf > 0)*2-1 ;
        pred_final(to_clf)=pred_clf(to_clf);
        accu = sum(pred_final==y)/N;
        if options.verbose
            cur_task = getCurrentTask(); 
            if isempty(cur_task)
                disp(['EM iter ',num2str(em_iter),': frac to partial=',num2str(sum(to_clf)/N),' | accu=', num2str(accu), ' q1 diff=', num2str(norm(q1-q1_old,1)), ' accu diff=', num2str(abs(accu-accu_old))]);
            else
                disp([num2str(cur_task.ID),' EM iter ',num2str(em_iter),': frac to partial=',num2str(sum(to_clf)/N),' | accu=', num2str(accu), ' q1 diff=', num2str(norm(q1-q1_old,1)), ' accu diff=', num2str(abs(accu-accu_old))]);
            end
        end
        % check whether to stop EM early
        if abs(accu-accu_old)<options.accu_threshold && norm(q1-q1_old,1)<options.q_threshold
            if options.verbose
                disp(['EM iter early exited! thresholds met!']);
            end
            break;
        end
        q1_old = q1;
        accu_old = accu;
        save(['ens_',options.output_file], 'ensemble_gate', 'ensemble_clf');
    end %end of EM iterations
end

function [q1, q0] = updateQ(p_gate, p_clf, p_full, y,proba_train,N)
    pz1 = 1./(1+exp(-p_gate));
    pz0 = 1 - pz1;

    pyz0 = 1./(1+exp(-y.*p_clf));

    w1 = proba_train.*pz1;
    w0 = pyz0.*pz0;
    % I-projection to satisfy the posterior constraint
    % use binary search for gamma
    g_max_cur = 1e5;
    g_min_cur = 1e-5;
    gamma_threashold = 1e-3;
    max_gamma_iter = 100;

    for gamma_iter = 1:max_gamma_iter
        gamma = 0.5*(g_max_cur+g_min_cur);
        frac_cur = sum(w1.*gamma./(w1*gamma+w0));
        if frac_cur > p_full*N
            g_max_cur = gamma;
        else
            g_min_cur = gamma;
        end
        if abs(frac_cur-p_full*N) < gamma_threashold
            break
        end
    end
    Z = w1*gamma + w0;
    q1 = w1*gamma./Z;
    q0 = w0./Z;

    if any(isnan(q1)) || any(isnan(q0))
        error('q0 or q1 is nan');
    end
end
