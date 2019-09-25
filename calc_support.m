function X_sampled = calc_support(X,K,c,smp_type,ker_params)

% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% Subsample c columns from the train set X, or the kernel matrix K, 
% using a sampling method given by smp_method.
%
% INPUT:
% X          - data matrix
% K          - kernel matrix of data: K(X,X)
% c          - number of samples for Nystrom
% smp_method - sampling method: 'uniform'/'kmeans'/'coreset'/'diag'/'col_norm'
% ker_params - structure containing kernel parameters
%
% OUTPUT:
% X_sampled  - sampled dataset of X
% ========================================================================

N = size(X,2);
if (c == N)          % trivial case where c = N
    X_sampled = X;
    return
end
switch smp_type
    case 'kmeans'
        [~,X_sampled,~] = fkmeans(X',c);
        X_sampled = X_sampled';
    case 'coreset'
        coresetObj = KSVDCoresetAlg();
        coresetObj.sampleSize = c;
        coresetObj.svdVecs = 0;
        [X_sampled,~,~,~] = computeCoreset(coresetObj,X');
        X_sampled = X_sampled';
    case 'uniform'
        permute_vec = randperm(N);
        supp = permute_vec(1:c);
        X_sampled = X(:,supp);
    case 'diag'
        if (strcmp(ker_params.ker_type,'Gaussian'))
            permute_vec = randi(N,[1,N]);
            supp = permute_vec(1:c);
            X_sampled = X(:,supp);
        else
            Gii = sum(X.^2);
            if (strcmp(ker_params.ker_type,'Linear'))
                p = Gii/sum(Gii);
            elseif (strcmp(ker_params.ker_type,'Polynomial'))
                Kii = (ker_params.ker_param_2 + Gii).^(ker_params.ker_param_1);
                p = Kii/sum(Kii);
            end
            px = 1:N;
            supp = round(randpdf(p,px,[1,c]));
            X_sampled = X(:,supp);
        end
    case 'col_norm'
        col_norm = sum(K.^2);
        p = col_norm/sum(col_norm);
        px = 1:N;
        supp = round(randpdf(p,px,[1,c]));
        supp = supp(~isnan(supp));
        X_sampled = X(:,supp);
end