function dic_cell = init_dictionary(classify_params,train_cell)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
% 
% Initialize dictionary before performing dictionary learning
%
% INPUT:
% classify_params - struct containing all classification params
% train_cell      - cell containing the train input divided by classes
%
% OUTPUT:
% dic_cell        - cell containing a dictionary initialization for each
% class
% ========================================================================

init_dic       = classify_params.init_dic;
alg_type       = classify_params.alg_type;
num_runs       = classify_params.num_runs;
num_classes    = classify_params.num_classes;
num_atoms      = classify_params.num_atoms;
ker_type       = classify_params.ker_type;
ker_param_1    = classify_params.ker_param_1;
ker_param_2    = classify_params.ker_param_2;
sig_dim        = size(train_cell{1},1);
dic_cell       = cell(1,num_classes);

switch alg_type
    case {'KSVD','RLS'}
        switch init_dic
            case 'random' % entirely random elements
                for i = 1:num_classes
                    dic_cell{i} = randn(sig_dim,num_atoms);
                end
            case 'partial' % initial columns in dictionary are some of the train samples
                for i = 1:num_classes
                    if (num_runs == 1)
                        rng(i);
                    end
                    if ((size(train_cell{i},2)) >= num_atoms)
                        ind = randperm(size(train_cell{i},2));
                        ind = ind(1:num_atoms);
                    else
                        ind = randperm(num_atoms);
                        ind = mod(ind,size(train_cell{i},2)) + 1;
%                         ind = ind(1:num_atoms);
                    end
                    dic_cell{i} = train_cell{i}(:,ind) ;
                    dic_cell{i} = dic_cell{i}.*repmat(1./sqrt(sum(dic_cell{i}.*dic_cell{i})),[sig_dim,1]);
                end
        end
    case 'KKSVD'
        switch init_dic
            case 'random'
                for i = 1:num_classes
                    dic_cell{i} = randn(size(train_cell{i},2),num_atoms);
                end
            case 'partial'
                for i = 1:num_classes
                    if (num_runs == 1)
                        rng(i);
                    end
                    dic_cell{i} = zeros(size(train_cell{i},2),num_atoms);
                    ind = randperm(size(train_cell{i},2));
                    ind = ind(1:num_atoms);
                    for j = 1:num_atoms
                        ker_params = struct('X',0,'Y',0,...
                            'ker_type',ker_type,'ker_param_1',ker_param_1,'ker_param_2',ker_param_2);
                        atom_ker_norm = calc_kernel(train_cell{i}(:,ind(j))'*train_cell{i}(:,ind(j)),ker_params);
                        dic_cell{i}(ind(j),j) = 1/sqrt(atom_ker_norm);
                    end
                    dic_cell{i} = dic_cell{i}.*repmat(1./sqrt(sum(dic_cell{i}.*dic_cell{i},2)),[1,num_atoms]);
                end
        end
end