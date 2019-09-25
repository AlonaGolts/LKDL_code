% ========================================================================
% Demo: running LC-KSVD algorithm with and without LKDL
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 06-04-2016
% ========================================================================

clear all clear all;
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP
load AR

%% define params for LC-KSVD with LKDL

params.linear_approx      = 1;                       % perform LKDL pre-processing yes (1) or no (0)
params.ker_type           = 'Gaussian';              % type of kernel
params.ker_param_1        = 0.5;                     % first kernel parameter (see calc_kernel)
params.ker_param_2        = 0;                       % second kernel parameter (ususally zero)
params.c                  = size(train_img,2);       % number of sampled columns in the training dataset
params.k                  = 540;                     % eigendecomposition approximation parameter
params.samp_method        = 'uniform';               % sampling method for Nystrom's method
params.sparsitythres      = 30;                      % sparsity prior
params.sqrt_alpha         = 1/700;                   % weights for label constraint term 
params.sqrt_beta          = 1/30;                    % weights for classification err term  
params.dictsize           = 500;                     % dictionary size
params.iterations         = 50;                      % iteration number
params.iterations4ini     = 20;                      % iteration number for initialization
params.num_runs           = 1;                       % number of runs

%% normalize input set before applying kernel

dim = size(train_img,1);
train_img = train_img./repmat(sqrt(sum(train_img.^2)),[size(train_img,1) 1]);
test_img = test_img./repmat(sqrt(sum(test_img.^2)),[size(test_img,1) 1]);

%% perform LC-KSVD1 with LKDL

params.sqrt_alpha = 1/14;
params.sqrt_beta = 1/14;

[accuracy1,~,~,~] = LCKSVD_aux(train_img, test_img, train_lbl, test_lbl, params);
fprintf('\nFinal recognition rate for LC-KSVD1 + LKDL is : %.04f ', accuracy1);

%% perform LC-KSVD2 with LKDL

params.sqrt_alpha = 1/15;
params.sqrt_beta = 1/17;

[~,accuracy2,~,~] = LCKSVD_aux(train_img, test_img, train_lbl, test_lbl, params);
fprintf('\nFinal recognition rate for LC-KSVD2 + LKDL is : %.04f ', accuracy2);
