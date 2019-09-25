function [train_cell,test_images,dic_cell,A_cell,B_cell,train_t] = RLS_train(classify_params,dic_cell,A_cell,B_cell)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% INPUT:
% classify_params - struct containing all parameters for classification.
% dic_cell        - optional initialization for dictionary
% A_cell          - cell containing current A matrix in 'dictlearn_mb_simple'
% B_cell          - cell containing current B matrix in 'dictlearn_mb_simple'
%
% OUTPUT:
% train_cell      - virtual train set divided to classes
% test_images     - virtual test set
% dic_cell        - trained dictionary divided to classes
% A_cell          - updated value of A matrix in 'dictlearn_mb_simple'
% B_cell          - updated value of B matrix in 'dictlearn_mb_simple'
% train_t         - total time of training (for entire database)
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
lam0                  = classify_params.lam0;              % initial value of lam in mini-batch-RLS
minibatch             = classify_params.minibatch;         % minibatch sizes in mini-batch-RLS

train_cell = cell(1,num_classes);
% divide the training set to different classes
for i = 1:num_classes
    train_cell{i} = train_images(:,train_labels(i,:) == 1);
end

% initialize dictionary
if isempty(dic_cell{1})
    dic_cell = init_dictionary(classify_params,train_cell);
end
% initialize matrix A for mini-batch RLS-DLA
if isempty(A_cell{1})
    for i = 1:num_classes
        A_cell{i} = eye(num_atoms);
    end
end

if isempty(B_cell{1})
    B_cell = dic_cell;
end

% dictionary training
train_tic = tic;
h = waitbar(0,'Training Dictionary');
for i = 1:num_classes

    params = [];
    params.X = train_cell{i};
    params.samet = 'javaormp';
    params.saopt = struct('tnz',card,'verbose',0);
    params.K = num_atoms;
    params.D = dic_cell{i};
    params.A = A_cell{i};
    params.B = B_cell{i};
    params.verbose = 0;
    params.minibatch = minibatch;
    params.lam0 = lam0;
    Ds = dictlearn_mb_simple(params);
    dic_cell{i} = Ds.D;
    A_cell{i} = Ds.A;
    B_cell{i} = Ds.B;
    
    waitbar(i/num_classes);
end
close(h);
train_t = toc(train_tic);