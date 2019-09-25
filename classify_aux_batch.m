function [results] = classify_aux_batch(params)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% INPUT:
% params  - struct containing all parameters for classification.
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
    
    total_time = tic;
    
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
            test = test - repmat(mean(test),[size(test,1),1]);
            test = test./repmat(sqrt(sum(test.^2)),[size(test,1),1]);
        case 'none'
    end
    
    % sample training set for using Nystrom method later on and compute
    % reduced kernel matrix W 
    if (params.linear_approx == 1) 
        virtual_train_t_1 = tic;
        train_R = [];
        c = round(params.c/params.num_batches);
        for i = 1:params.num_batches
            DB_path = [pwd,'\databases\',params.database,num2str(params.num_batches),'_',num2str(i)];
            load(DB_path);
            ker_params.ker_type = params.ker_type;
            ker_params.ker_param_1 = params.ker_param_1;
            ker_params.ker_param_2 = params.ker_param_2;
            if strcmp(params.samp_method,'col_norm')
                ker_params.X = train_img;
                ker_params.Y = train_img;
                K = calc_kernel(train_img'*train_img,ker_params);
                train_R = [train_R, calc_support(train_img,K,c,params.samp_method,ker_params)]; %% subsample of training
            else
                train_R = [train_R, calc_support(train_img,0,c,params.samp_method,ker_params)]; %% subsample of training
            end
        end
        ker_params.X = train_R;
        ker_params.Y = train_R;
        ker_params.ker_type = params.ker_type;
        ker_params.ker_param_1 = params.ker_param_1;
        ker_params.ker_param_2 = params.ker_param_2;
        W = calc_kernel(train_R'*train_R,ker_params);
        [V,D] = eigs(W,params.k,'la');
        virtual_train_t(nn) = virtual_train_t(nn) + toc(virtual_train_t_1);
        % the time needed to compute the eigen-decomposition of W
        disp(['virtual_train_1: ',num2str(toc(virtual_train_t_1))]);
    end

    classify_params.test_images = test;
    classify_params.test_labels = test_l;
    
%     total_time = tic;
    switch classify_params.alg_type
        case {'KSVD','RLS'}
            dic_cell = cell(params.num_classes,1);
            A_cell = cell(params.num_classes,1); % data structure for mini-batch RLS-DLA algorithm
            B_cell = cell(params.num_classes,1); % data structure for mini-batch RLS-DLA algorithm
            for i = 1:params.num_batches
                DB_path = [pwd,'\databases\',params.database,num2str(params.num_batches),'_',num2str(i)];
                load(DB_path);
                
                switch (params.pre_process)
                    case 'mean_std'
                        train_img = train_img - repmat(mean(train_img),[size(train_img,1),1]);
                        train_img = train_img./repmat(sqrt(sum(train_img.^2)),[size(train_img,1),1]);
                    case 'none'
                end
                
                classify_params.train_images = train_img;
                classify_params.train_labels = train_lbl;
                classify_params.train_size = size(train_img,2);
                
                % calculate relative part of C_i corresponding with X_i 
                if (params.linear_approx == 1)
                    virtual_train_t_2 = tic;
                    ker_params.X = train_img;
                    ker_params.Y = train_R;
                    ker_params.ker_type = params.ker_type;
                    ker_params.ker_param_1 = params.ker_param_1;
                    ker_params.ker_param_2 = params.ker_param_2;
                    C_i = calc_kernel(train_img'*train_R,ker_params);
                    classify_params.train_images = pinv(D.^(1/2))*V'*C_i';
                    virtual_train_t(nn) = virtual_train_t(nn) + toc(virtual_train_t_2);
                    % the time needed to perform matrix multiplication for
                    % each mibni-batch in Nystrom
                    disp(['virtual_train_2: ',num2str(toc(virtual_train_t_2))]);
                end
                
                [train_cell,test,dic_cell,A_cell,B_cell,train_time] = RLS_train(classify_params,dic_cell,A_cell,B_cell);
                train_t(nn) = train_t(nn) + train_time;
                disp(['train time: ',num2str(train_t(nn))]);
                disp(['Batch #',num2str(i),' done!']);
            end
                
            if (params.linear_approx == 1)
                virtual_test_time = tic;
                ker_params.X = test;
                ker_params.Y = train_R;
                ker_params.ker_type = params.ker_type;
                ker_params.ker_param_1 = params.ker_param_1;
                ker_params.ker_param_2 = params.ker_param_2;
                test_C = calc_kernel(test'*train_R,ker_params);
                test = pinv(D.^(1/2))*V'*test_C';
                virtual_test_t(nn) = toc(virtual_test_time);
                % the time needed to create the virtual test set
                disp(['virtual_test: ',num2str(virtual_test_t(nn))]);
            end
            
            [percent(nn), classify_results, sp_codes, classify_t(nn)] = RLS_classify(classify_params,train_cell,test,dic_cell);
            disp(['test time: ',num2str(classify_t(nn))]);
            
        case 'KKSVD'
            [train_cell,K_YY_cell,test,dic_cell,train_t(nn)] = KKSVD_train(classify_params);
            [percent(nn), classify_results, sp_codes, classify_t(nn)] = KKSVD_classify(classify_params,train_cell,K_YY_cell,test,dic_cell);
    end
    total_t(nn) = toc(total_time);
%     disp(['total_time: ',num2str(total_t(nn))]);
    disp([num2str(nn),' out of ',num2str(params.num_runs), ', Accuracy: ',num2str(percent(nn))]);
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