function [loss, gate_der, clf_der] = lossNgrad_gate_clf(y,p_gate, p_clf, q0,q1)
    ee_gate = exp(p_gate);
    ee_clf = exp(-y.*p_clf);
    N = length(y);
    loss = (q1'*log(1+1./ee_gate)+ q0'*(log(1+ee_clf) + log(1+ee_gate)));

    gate_der = - (- q1 ./ (1+ee_gate) + q0 ./ (1+ 1./ee_gate)) ; % negative of gradient
    clf_der =  - (-y.*q0./ (1+1./ee_clf)); % negative of gradient
