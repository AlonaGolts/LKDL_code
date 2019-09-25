function [accuracy, classify_results, sp_codes, classify_t] = KKSVD_classify(classify_params,train_cell,K_YY_cell,test,dic_cell)

% ========================================================================
% Based on the code given by Nguyen et al.
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% INPUT:
% classify_params  - struct containing all parameters for classification.
% train_cell       - virtual train set divided to classes
% test             - virtual test set
% dic_cell         - trained dictionary divided to classes
%
% OUTPUT:
% accuracy         - total accuracy of classification
% classify_results - strurcture containing the calculated labels (to compare with original test labels)
% classify_t       - total time of testing (for entire database)
% ========================================================================

num_classes = classify_params.num_classes;
ker_type = classify_params.ker_type;
ker_param_1 = classify_params.ker_param_1;
ker_param_2 = classify_params.ker_param_2;
card = classify_params.card;
X = cell(num_classes,1);
res = zeros(num_classes,size(test,2));
classify_results = zeros(num_classes,size(test,2));
classify_tic = tic;
h = waitbar(0,'Classifying Test Examples');
for i = 1:num_classes
    mynorm = Knorms(eye(size(K_YY_cell{i},1)),K_YY_cell{i}) ;
    mynorm = mynorm(:) ;
    K_YY_cell{i} = K_YY_cell{i}./(mynorm*mynorm')  ; % normalize to norm-1 in feature space
    K_ZY = gram(test', train_cell{i}',ker_type,ker_param_2,ker_param_1) ;
    K_ZY = K_ZY./(repmat(mynorm',size(K_ZY,1),1) );
    K_ZZ = gram(test',test',ker_type,ker_param_2,ker_param_1) ;
    [X{i}, res(i,:)] = KOMP(dic_cell{i},K_YY_cell{i},K_ZY,card,K_ZZ) ;
    waitbar(i/num_classes);
end
close(h);

[~,min_ind] = min(res,[],1);
lin_ind = sub2ind(size(res),min_ind,1:size(res,2));
classify_results(lin_ind) = 1;
diff = sum(abs(classify_results - classify_params.test_labels));
accuracy = sum(diff==0)/length(diff);
% disp([classify_params.alg_type,': ',num2str(accuracy)]);
classify_t = toc(classify_tic);    %% classifcation time
sp_codes = 0;