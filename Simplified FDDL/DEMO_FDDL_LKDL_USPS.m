% ========================================================================
% Demo: running FDDL algorithm with and without LKDL
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 06-04-2016
% ========================================================================

clear all clear all;
load USPS
[train_lbl,~] = find(train_lbl == 1);
train_lbl = train_lbl';
[test_lbl,~] = find(test_lbl == 1);
test_lbl = test_lbl';
train_size = size(train_img,2);

%% define params for FDDL without LKDL
params.linear_approx      = 0;                              % perform LKDL pre-processing yes (1) or no (0)
params.ker_type           = 'Polynomial';                   % type of kernel
params.ker_param_1        = 3;                              % first kernel parameter (see calc_kernel)
params.ker_param_2        = 0;                              % second kernel parameter (ususally zero)
params.c                  = round(0.2*size(train_img,2));   % number of sampled columns in the training dataset
params.k                  = 256;                            % eigendecomposition approximation parameter
params.samp_method        = 'kmeans';                       % sampling method for Nystrom's method
params.dic_ini            = 'kmeans';                       % initialization method of dictionary
params.atoms_per_dic      = 300;                            % number of atoms in each sub-dictionary
params.num_runs           = 5;                              % number of runs
params.pre_process        = 'mean_std';

%%
switch (params.pre_process)
    case 'mean_std'
        train_img = train_img - repmat(mean(train_img),[size(train_img,1),1]);
        train_img = train_img./repmat(sqrt(sum(train_img.^2)),[size(train_img,1),1]);
        test_img = test_img - repmat(mean(test_img),[size(test_img,1),1]);
        test_img = test_img./repmat(sqrt(sum(test_img.^2)),[size(test_img,1),1]);
    case 'none'
    case 'std'
        train_img = train_img./repmat(sqrt(sum(train_img.^2)),[size(train_img,1),1]);
        test_img = test_img./repmat(sqrt(sum(test_img.^2)),[size(test_img,1),1]);
end

%% perform FDDL without LKDL

values = 1;
for k = 1:length(values)
    acc_accuracy = 0;
    for i = 1:params.num_runs
        [accuracy] = FDDL_aux(train_img, train_lbl, test_img, test_lbl, params);
        acc_accuracy = acc_accuracy + accuracy;
    end
    accuracy(k) = acc_accuracy/params.num_runs;
    fprintf('\nFinal recognition rate for FDDL is : %.03f ', accuracy(k));
end
FDDL = accuracy;

%% define params for FDDL without LKDL
params.linear_approx      = 1;                           % perform LKDL pre-processing yes (1) or no (0)
params.ker_type           = 'Polynomial';                % type of kernel
params.ker_param_1        = 2;                           % first kernel parameter (see calc_kernel)
params.ker_param_2        = 0;                           % second kernel parameter (ususally zero)
params.c                  = round(0.2*size(train_img,2));   % number of sampled columns in the training dataset
params.k                  = 256;                         % eigendecomposition approximation parameter
params.samp_method        = 'kmeans';                    % sampling method for Nystrom's method
params.dic_ini            = 'kmeans';                    % initialization method of dictionary
params.atoms_per_dic      = 300;                         % number of atoms in each sub-dictionary
params.num_runs           = 5;                          % number of runs

%% perform FDDL with LKDL
values = 1;
for k = 1:length(values)
    acc_accuracy = 0;
    for i = 1:params.num_runs
        [accuracy] = FDDL_aux(train_img, train_lbl, test_img, test_lbl, params);
        acc_accuracy = acc_accuracy + accuracy;
    end
    accuracy(k) = acc_accuracy/params.num_runs;
    fprintf('\nFinal recognition rate for FDDL + LKDL kmeans is : %.03f ', accuracy(k));
end
FDDL_LDKL_kmeans = accuracy;

