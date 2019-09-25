function [accuracy, classify_results, sp_codes, classify_t] = RLS_classify(classify_params,train_cell,test,dic_cell)

% ========================================================================
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
% spcodes          - optional, if one would like to return the sparse codes
% classify_t       - total time of testing (for entire database)
% ========================================================================

num_classes = classify_params.num_classes;
X = cell(num_classes,1);
res = zeros(num_classes,size(test,2));
classify_results = zeros(num_classes,size(test,2));
classify_tic = tic;
h = waitbar(0,'Classifying Test Examples');
for i = 1:num_classes
    saopt = struct('tnz',classify_params.card,'verbose',0);
%     X{i} = sparseapprox(test, dic_cell{i}, 'javaORMP', saopt);
    X{i} = omp(dic_cell{i}'*test, dic_cell{i}'*dic_cell{i}, classify_params.card);
    res(i,:) = sqrt(sum((test - dic_cell{i}*X{i}).^2));
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