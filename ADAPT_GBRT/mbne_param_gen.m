% generate parameter file for yahoo em+greedymiser
Lambda = [logspace(-1, 1,8), logspace(1.2,3,8)];
LearningRate = linspace(0.5,1,6);
P_full = linspace(.2,.9, 15);
nTrees = 100;
max_em_iter = 50;
interval  = 10;
depth = 4;

[x1, x2, x3, x4, x5, x6, x7] = ndgrid(Lambda, LearningRate, P_full, nTrees, max_em_iter, interval, depth);
param = [x1(:), x2(:), x3(:), x4(:), x5(:), x6(:), x7(:)];

dlmwrite('mbne_lstsqparam1',param);
