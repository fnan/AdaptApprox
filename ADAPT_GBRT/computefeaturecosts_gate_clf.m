function [featurecosts] = computefeaturecosts_gate_clf(lambda, defaultcosts,ensemble_gate, ensemble_clf)
    alltrees_gate = vertcat(ensemble_gate{2}{:});
    alltrees_clf = vertcat(ensemble_clf{2}{:});
    alltrees = [alltrees_gate; alltrees_clf];
	if length(alltrees) > 0
		usedfeatures = unique(alltrees(:,1));
		usedfeatures = usedfeatures(usedfeatures>0);
		defaultcosts(usedfeatures) = 0;
	end
	featurecosts = lambda * defaultcosts;
end