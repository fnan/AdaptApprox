function [args] = preprocesslayer_sqrimpurity(data, options) 
	% confirm necessary data or options
	assert(isfield(data,'numfeatures'))
	assert(isfield(data,'n'));
	assert(isfield(data,'y'));
	assert(isfield(data,'depth'));
	assert(isfield(data,'tree'))
	
	% compute counts for each node
	numnodes = 2^(data.depth-1);
	for i=1:numnodes,
		m_infty(i) = sum(data.n==i);
		l_infty(i) = sum(data.y(data.n==i));
	end
	
	% get parents
	parents = getlayer(data.tree,data.depth);
	
	% include feature costs
	if isfield(options,'featurecosts'),
		featurecosts = options.featurecosts;
	else,
		featurecosts = zeros(data.numfeatures,1);
	end
	
	% create impurity evalution argument
	args = {m_infty, l_infty, parents(:,4), featurecosts};
end
