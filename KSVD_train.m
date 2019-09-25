function [train_cell,test_images,dic_cell,train_t] = KSVD_train(classify_params)
% function [train_cell,test_images,dic_cell,train_t] = KSVD_train(classify_params,dic_cell)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% INPUT:
% classify_params - struct containing all parameters for classification.
%
% OUTPUT:
% train_cell      - virtual train set divided to classes
% test_images     - virtual test set
% dic_cell        - trained dictionary divided to classes
% train_t         - total time of training (for entire database)
% virtual_train_t - total time it takes to compute the virtual train set
% virtual_test_t  - total time it takes to compute the virtual test set
% ========================================================================

% Parameters in classify_params
train_images          = classify_params.train_images;      % training examples
test_images           = classify_params.test_images;       % training labels
test_labels           = classify_params.test_labels;       % test examples
train_labels          = classify_params.train_labels;      % test labels
num_classes           = classify_params.num_classes;       % number of classes in database
alg_type              = classify_params.alg_type;          % algortihm: 'KSVD','KKSVD'
init_dic              = classify_params.init_dic;          % initializtion method of dictionary
num_atoms             = classify_params.num_atoms;         % number of atoms in each class' dictionary
iter                  = classify_params.iter;              % number of dictionary learning iterations
card                  = classify_params.card;              % cardinality of sparse representations

train_cell = cell(1,num_classes);
% divide the training set to different classes
for i = 1:num_classes
    train_cell{i} = train_images(:,train_labels(i,:) == 1);
end

% initialize dictionary
dic_cell = init_dictionary(classify_params,train_cell);

% dictionary training
train_tic = tic;
h = waitbar(0,'Training Dictionary');
for i = 1:num_classes
    params = [];
    params.data = train_cell{i};
    params.Tdata = card;
    params.iternum = iter;
    params.initdict = dic_cell{i};
    params.memusage = 'high';
    dic_cell{i} = ksvd(params);
    waitbar(i/num_classes);
end
close(h);
train_t = toc(train_tic);