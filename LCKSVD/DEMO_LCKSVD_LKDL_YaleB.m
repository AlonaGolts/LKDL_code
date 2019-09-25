% ========================================================================
% Demo: running LC-KSVD algorithm with and without LKDL
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 06-04-2016
% ========================================================================

clear all clear all;
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP
load('featurevectors');

%% define params for LC-KSVD without LKDL

params.linear_approx      = 0;                       % perform LKDL pre-processing yes (1) or no (0)
params.ker_type           = 'Gaussian';              % type of kernel
params.ker_param_1        = 1;                       % first kernel parameter (see calc_kernel)
params.ker_param_2        = 0;                       % second kernel parameter (ususally zero)
params.c                  = size(training_feats,2);  % number of sampled columns in the training dataset
params.k                  = 504;                     % eigendecomposition approximation parameter
params.samp_method        = 'kmeans';                % sampling method for Nystrom's method
params.sparsitythres      = 30;                      % sparsity prior
params.sqrt_alpha         = 4;                       % weights for label constraint term 
params.sqrt_beta          = 2;                       % weights for classification err term  
params.dictsize           = 570;                     % dictionary size
params.iterations         = 50;                      % iteration number
params.iterations4ini     = 20;                      % iteration number for initialization
params.num_runs           = 1;                       % number of runs

% perform LC-KSVD without LKDL

values = 1;
for k = 1:length(values)
    acc_accuracy1 = 0;
    acc_accuracy2 = 0;
    for i = 1:params.num_runs
        [accuracy1,accuracy2,LCKSVD_sp_codes1,LCKSVD_sp_codes2] = LCKSVD_aux(training_feats, testing_feats, H_train, H_test, params);
        acc_accuracy1 = acc_accuracy1 + accuracy1;
        acc_accuracy2 = acc_accuracy2 + accuracy2;
    end
    accuracy1(k) = acc_accuracy1/params.num_runs;
    accuracy2(k) = acc_accuracy2/params.num_runs;
    fprintf('\nFinal recognition rate for LC-KSVD1 is : %.03f ', accuracy1(k));
    fprintf('\nFinal recognition rate for LC-KSVD2 is : %.03f \n', accuracy2(k));
end


%% define params for LC-KSVD without LKDL

params.linear_approx      = 1;                       % perform LKDL pre-processing yes (1) or no (0)
params.ker_type           = 'Gaussian';              % type of kernel
params.ker_param_1        = 2;                       % first kernel parameter (see calc_kernel)
params.ker_param_2        = 0;                       % second kernel parameter (ususally zero)
params.c                  = size(training_feats,2);  % number of sampled columns in the training dataset
params.k                  = 504;                     % eigendecomposition approximation parameter
params.samp_method        = 'kmeans';                % sampling method for Nystrom's method
params.sparsitythres      = 30;                      % sparsity prior
% params.sqrt_alpha       = 4;                       % weights for label constraint term 
% params.sqrt_beta        = 2;                       % weights for classification err term  
params.sqrt_alpha         = 1/1000;
params.sqrt_beta          = 1/1000;
params.dictsize           = 570;                     % dictionary size
params.iterations         = 50;                      % iteration number
params.iterations4ini     = 20;                      % iteration number for initialization
params.num_runs           = 1;                       % number of runs

%% normalize input set before applying kernel

dim = size(training_feats,1);
training_feats = training_feats./repmat(sqrt(sum(training_feats.^2)),[size(training_feats,1) 1]);
testing_feats = testing_feats./repmat(sqrt(sum(testing_feats.^2)),[size(testing_feats,1) 1]);

%% LCKSVD1 with LKDL
params.sqrt_alpha = 1/200;
params.sqrt_beta = 1/200;

[accuracy1,~,~,~] = LCKSVD_aux(training_feats, testing_feats, H_train, H_test, params);
fprintf('\nFinal recognition rate for LC-KSVD1 + LKDL is : %.04f ', accuracy1);

%% LCKSVD2 with LKDL
params.sqrt_alpha = 1/600;
params.sqrt_beta = 1/900;

[~,accuracy2,~,~] = LCKSVD_aux(training_feats, testing_feats, H_train, H_test, params);
fprintf('\nFinal recognition rate for LC-KSVD2 + LKDL is : %.04f \n', accuracy2);