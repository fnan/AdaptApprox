function[loss,gradient]=logisticloss(y,p)   
    ee = exp(-y.*p);
    loss=sum(log(1+ee));
    gradient= y./(1+1./ee);
end 
