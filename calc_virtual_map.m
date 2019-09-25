function [train_map, test_map, virtual_train_t, virtual_test_t] = calc_virtual_map(train, test, ker_params,smp_type,c,k)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% calculate the virtual train and test sets for LKDL
%
% INPUT:
% train           - input train set
% test            - input test set
% ker_params      - struct containing kernel parameters
% smp_type        - sampling method - 'uniform'/'kmeans'/'coreset'/'diag'/'col-norm'
% c               - number of samples for Nystrom
% k               - desired dimension after eigen-decomposition
%
% OUTPUT:
% virtual_train_t - time it takes to compute the virtual train set
% virtual_test_t  - time it takes to compute the virtual test set
% train_map       - train virtual samples
% test_map        - test virtual samples
% ========================================================================

% in case of col_norm sampling, the kernel matrix is needed
train_t = tic;
if strcmp(smp_type,'col_norm')
    ker_params.X = train;
    ker_params.Y = train;
    K = calc_kernel(train'*train,ker_params);
    Y = calc_support(train,K,c,smp_type,ker_params); %% subsample of training
else
    Y = calc_support(train,0,c,smp_type,ker_params); %% subsample of training
end

ker_params.X = train;
ker_params.Y = Y;
C = calc_kernel(train'*Y,ker_params);
ker_params.X = Y;
W = calc_kernel(Y'*Y,ker_params);
if (c == k)
    [V,D] = eig(W);
else
    [V,D] = eigs(W,k,'la');
end
FT = C*V*pinv(D.^(1/2));
train_map = FT';

virtual_train_t = toc(train_t);
test_t = tic;
ker_params.X = test;
ker_params.Y = Y;
test_C = calc_kernel(test'*Y,ker_params);
test_map = (test_C*V*pinv(D.^(1/2)))';
virtual_test_t = toc(test_t);