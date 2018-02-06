%% Builds a collection of limited-depth regression trees.
%% Returns a cell array of trees and the loss value after training.

function [ensemble,loss,p] = gbrt(X,lossNgrad,options)
	% check for required arguments and outputs
	if nargin < 2,
		error('Too few arguments');
	elseif nargin > 3,
		error('Too many arguments');
	end
	
	% verify lossNgrad
	assert(isa(lossNgrad,'function_handle'),...
		'Argument lossNgrad must be function handle returning loss and gradient for predictions');
	
	% if necessary, instantiate options
	if nargin == 2,
		options = struct;
	else
		assert(isa(options,'struct'),'Argument options must be a struct');
	end
	
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
	
	% initialize ensemble (cell array of trees)
	ensemble = {[],{}};
	
	% initialize predictions
	if isfield(options,'initpreds'),
		%assert(all(size(options.initpreds) == [N,1]),...
		%	'options.initpreds must be a vector of length size(X,1)');
		p = options.initpreds;
	else
		p = zeros(N,1);
	end
	
	% compute initial loss and gradient
	loss = [];
	[loss(1),g] = lossNgrad(p); % TODO include ensemble or other args
		% loss is the loss function value
    	% g is the gradient w.r.t. each instance
	if options.verbose, fprintf('Initial loss %f\n',loss(end)); end;
	
	% construct trees to minimize loss
    for t=1:options.ntrees,
		% compute feature costs
		if isfield(options,'computefeaturecosts')
			options.featurecosts = options.computefeaturecosts(ensemble);
		end
		% progressbar(t,options.ntrees,'Making Trees')
		% construct tree(s)
		% try
			if size(g,2) == 1, % construct one tree for vector gradient
				[tree,pt] = buildtree(X,Xs,Xi,g,options.depth,options);
			else, % construct multiple trees for multidimensional gradient
				tree = {};
				pt = zeros(size(g));
				for i=1:size(g,2),
					[tree{i},pt(:,i)] = buildtree(X,Xs,Xi,g(:,i),options.depth,options);
				end
			end
		% catch
			% fprintf('Error while constructing tree %d. Returning.\n', t);
			% return
		% end
		% update predictions and ensemble
		p(:) = p(:) + options.learningrate(t) * pt(:); % update predictions
		ensemble{1}(t) = options.learningrate(t); % add learning rate to ensemble
		ensemble{2}{t} = tree; % add tree to ensemble
		[loss(t+1),g] = lossNgrad(p);
		if options.verbose, fprintf('After tree %d, loss %f\n', t, loss(end)); end;
    end
