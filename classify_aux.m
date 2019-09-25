function [results] = classify_aux(params)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% INPUT:
% classify_params  - struct containing all parameters for classification.
%
% OUTPUT:
% results - struct containing classification results and statistics
% ========================================================================

percent = zeros(params.num_runs,1);
train_t = zeros(params.num_runs,1);
virtual_train_t = zeros(params.num_runs,1);
virtual_test_t = zeros(params.num_runs,1);
classify_t = zeros(params.num_runs,1);
total_t = zeros(params.num_runs,1);
classify_params = params;

for nn = 1:params.num_runs
    
    train = params.train_images;
    train_l = params.train_labels;
    test = params.test_images;
    test_l = params.test_labels;
    
    % reduce training size if train_size < input training data
    if (params.train_size < size(train,2))
        permute_vec = randperm(size(train,2));
        supp = permute_vec(1:params.train_size);
        train = train(:,supp);
        train_l = train_l(:,supp);
    end
    
    % adding Gaussian noise with standard deviation: sigma
    if (params.sigma > 0)
        test = test + params.sigma*randn(size(test));
    end
    
    % adding missing pixel corruption
    if (params.missing_pixels > 0)
        for i = 1:size(test,2)
            ind = randperm(size(test,1));
            test(ind(1:ceil(params.missing_pixels*size(test,1))),i) = 0;
        end
    end
    
    % pre-processing
    switch (params.pre_process)
        case 'mean_std'
            train = train - repmat(mean(train),[size(train,1),1]);
            train = train./repmat(sqrt(sum(train.^2)),[size(train,1),1]);
            test = test - repmat(mean(test),[size(test,1),1]);
            test = test./repmat(sqrt(sum(test.^2)),[size(test,1),1]);
        case 'none'
        case 'std'
            train = train./repmat(sqrt(sum(train.^2)),[size(train,1),1]);
            test = test./repmat(sqrt(sum(test.^2)),[size(test,1),1]);
    end

    % calculate embedding
    switch (params.linear_approx) 
        case 0 % no embedding
            virtual_train_t(nn) = 0;
            virtual_test_t(nn) = 0;
        case 1 % LKDL
            ker_params = struct('X',0,'Y',0,'ker_type',params.ker_type,'ker_param_1',params.ker_param_1,'ker_param_2',params.ker_param_2);
            [train, test, virtual_train_t(nn), virtual_test_t(nn)] = ...
                calc_virtual_map(train,test,ker_params,params.samp_method,params.c,params.k);
        case 2 % RFF
            virtual_train_time = tic();
            w_train = randn(round(params.c/2), size(train,1));
            train = sqrt(round(params.c/2))*[cos(w_train*train); sin(w_train*train)];
            virtual_train_t(nn) = toc(virtual_train_time);
            virtual_test_time = tic();
            test = sqrt(round(params.c/2))*[cos(w_train*test); sin(w_train*test)];
            virtual_test_t(nn) = toc(virtual_test_time);
        case 3 % Fastfood
            virtual_train_time = tic();
            para = FastfoodPara(params.c, size(train,1));
            train = FastfoodForKernel(train, para, params.ker_param_1, 0, params.c);
            virtual_train_t(nn) = toc(virtual_train_time);
            virtual_test_time = tic();
            test = FastfoodForKernel(test, para, params.ker_param_1, 0, params.c);
            virtual_test_t(nn) = toc(virtual_test_time);
    end
    
    classify_params.train_images = train;
    classify_params.train_labels = train_l;
    classify_params.test_images = test;
    classify_params.test_labels = test_l;
    
    total_tic = tic;
    switch classify_params.alg_type
        case 'RLS'
            A_cell = cell(1,params.num_classes);
            B_cell = cell(1,params.num_classes);
            dic_cell = cell(1,params.num_classes);
            [train_cell,test,dic_cell,~,~,train_t(nn)] = RLS_train(classify_params,dic_cell,A_cell,B_cell);
            [percent(nn), classify_results, sp_codes, classify_t(nn)] = RLS_classify(classify_params,train_cell,test,dic_cell);
        case 'KSVD'
            [train_cell,test,dic_cell,train_t(nn)] = KSVD_train(classify_params);
            [percent(nn), classify_results, sp_codes, classify_t(nn)] = KSVD_classify(classify_params,train_cell,test,dic_cell);
        case 'KKSVD'
            [train_cell,K_YY_cell,test,dic_cell,train_t(nn)] = KKSVD_train(classify_params);
            [percent(nn), classify_results, sp_codes, classify_t(nn)] = KKSVD_classify(classify_params,train_cell,K_YY_cell,test,dic_cell);
    end
    total_t(nn) = toc(total_tic);
    disp([num2str(nn),' out of ',num2str(params.num_runs), ', Accuracy: ',num2str(percent(nn)), ', sig dim: ',num2str(size(train,1))]);
end

results.percent = mean(percent);                   % averaged accuracy result of classification
results.std = std(percent);                        % std of accuracy results over num_runs
results.virtual_train_t = mean(virtual_train_t);   % total time of computing virtual train set
results.virtual_test_t = mean(virtual_test_t);     % total time of computing virtual test set
results.train_t = mean(train_t);                   % total training time for entire database
results.classify_t = mean(classify_t);             % total test time for entire database
results.total_t = mean(total_t);                   % toal runtime: train_time + test_time + other_time(virtual samples)
results.class_vec = classify_results;              % resulting labels after classification
results.sp_codes = sp_codes;                       % sparse codes of test samples

disp(['Average accuracy: ',num2str(results.percent),', std: ',num2str(results.std)]);