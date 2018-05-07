function [featurecosts] = computefeaturecosts(lambda, defaultcosts, ensemble);
    alltrees = vertcat(ensemble{2}{:});
	if length(alltrees) > 0,
		usedfeatures = unique(alltrees(:,1));
		usedfeatures = usedfeatures(usedfeatures>0);
		defaultcosts(usedfeatures) = 0;
	end
	featurecosts = lambda * defaultcosts;
end