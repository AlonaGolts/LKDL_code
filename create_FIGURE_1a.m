% ========================================================================
% Create figure of Approximation accuracy vs. c/N (perent of samples for Nystrom)
% if you want to load the existing results, skip all the way to "show graphs"
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
% ========================================================================
 
%% choose 2000 random samples from USPS
load USPS
X = train_img;
dim = size(X,1);
N = 2000;
rand_vec = randperm(size(X,2));
X = X(:,rand_vec(1:N));
pre_process = 'mean_std';

%% pre-processing
switch pre_process
    case 'mean_std'
        X = X - repmat(mean(X),[size(X,1),1]);
        X = X./repmat(sqrt(sum(X.^2)),[size(X,1),1]);
    case 'none'
    case 'std'
        X = X./repmat(sqrt(sum(X.^2)),[size(X,1),1]);
end

%% parameters
ker_type = 'Polynomial';                      % Type of kernel: 'Polynomial'\'Gaussian'
ker_param_1 = 2;                              % first kernel paramter: order of 'Polynomial' or free parameter in 'Gaussian' 
ker_param_2 = 0;                              % second kernel parameter: scalar in 'Polynomial' kernel 
percent = 5:5:50;
c = round((1/100)*percent*N);                % number of samples in Nystrom approximation
k = dim;                                % dimension of signal after eigen-decomposition

%% construct kernel matrix
ker_params = struct('X',X,'Y',X,'ker_type',ker_type,'ker_param_1',ker_param_1,'ker_param_2',ker_param_2);
K = calc_kernel(X'*X,ker_params);

%% uniform approximation
norm_uni = zeros(length(percent),1);
for i = 1:length(c)
    ker_params.X = X;
    ker_params.Y = X;
    X_sampled = calc_support(X,K,c(i),'uniform',ker_params);
    ker_params.X = X;
    ker_params.Y = X_sampled;
    C = calc_kernel(X'*X_sampled,ker_params);
    ker_params.X = X_sampled;
    W = calc_kernel(X_sampled'*X_sampled,ker_params);
    if strcmp('eig','svd')
        [U,S,V] = svds(W,k);
        W_pinv = V*pinv(S)*U';
    else
        if (k ~= dim)
            [V,D] = eigs(W,k,'la');
        else
            [V,D] = eig(W);
        end
        W_pinv = V*pinv(D)*V';
    end
    K_uni = C*W_pinv*C';
    norm_uni(i) = norm(K - K_uni,'fro')/norm(K,'fro');
    disp(['uniform, c/N = ',num2str(percent(i)),...
        '% err = ',num2str(norm_uni(i))]);
end
disp('');

%% column-norm approximation
norm_colnorm = zeros(length(percent),1);
for i = 1:length(c)
    ker_params.X = X;
    ker_params.Y = X;
    X_sampled = calc_support(X,K,c(i),'col_norm',ker_params);
    ker_params.X = X;
    ker_params.Y = X_sampled;
    C = calc_kernel(X'*X_sampled,ker_params);
    ker_params.X = X_sampled;
    W = calc_kernel(X_sampled'*X_sampled,ker_params);
    if strcmp('eig','svd')
        [U,S,V] = svds(W,k);
        W_pinv = V*pinv(S)*U';
    else
        if (k ~= dim)
            [V,D] = eigs(W,k,'la');
        else
            [V,D] = eig(W);
        end
        W_pinv = V*pinv(D)*V';
    end
    K_colnorm = C*W_pinv*C';
    norm_colnorm(i) = norm(K - K_colnorm,'fro')/norm(K,'fro');
    disp(['colnorm, c/N = ',num2str(percent(i)),...
        '% err = ',num2str(norm_colnorm(i))]);
end
disp('');

%% diag approximation
norm_diag = zeros(length(percent),1);
for i = 1:length(c)
    ker_params.X = X;
    ker_params.Y = X;
    X_sampled = calc_support(X,K,c(i),'diag',ker_params);
    ker_params.X = X;
    ker_params.Y = X_sampled;
    C = calc_kernel(X'*X_sampled,ker_params);
    ker_params.X = X_sampled;
    W = calc_kernel(X_sampled'*X_sampled,ker_params);
    if strcmp('eig','svd')
        [U,S,V] = svds(W,k);
        W_pinv = V*pinv(S)*U';
    else
        if (k ~= dim)
            [V,D] = eigs(W,k,'la');
        else
            [V,D] = eig(W);
        end
        W_pinv = V*pinv(D)*V';
    end
    K_diag = C*W_pinv*C';
    norm_diag(i) = norm(K - K_diag,'fro')/norm(K,'fro');
    disp(['diag, c/N = ',num2str(percent(i)),...
        '% err = ',num2str(norm_diag(i))]);
end
disp('');

%% kmeans approximation
norm_kmeans = zeros(length(percent),1);
for i = 1:length(c)
    ker_params.X = X;
    ker_params.Y = X;
    X_sampled = calc_support(X,K,c(i),'kmeans',ker_params);
    ker_params.X = X;
    ker_params.Y = X_sampled;
    C = calc_kernel(X'*X_sampled,ker_params);
    ker_params.X = X_sampled;
    W = calc_kernel(X_sampled'*X_sampled,ker_params);
    if strcmp('eig','svd')
        [U,S,V] = svds(W,k);
        W_pinv = V*pinv(S)*U';
    else
        if (k ~= dim)
            [V,D] = eigs(W,k,'la');
        else
            [V,D] = eig(W);
        end
        W_pinv = V*pinv(D)*V';
    end
    K_kmeans = C*W_pinv*C';
    norm_kmeans(i) = norm(K - K_kmeans,'fro')/norm(K,'fro');
    disp(['kmeans, c/N = ',num2str(percent(i)),...
        '% err = ',num2str(norm_kmeans(i))]);
end
disp('');

%% coreset approximation
norm_coreset = zeros(length(percent),1);
for i = 1:length(c)
    ker_params.X = X;
    ker_params.Y = X;
    X_sampled = calc_support(X,K,c(i),'coreset',ker_params);
    ker_params.X = X;
    ker_params.Y = X_sampled;
    C = calc_kernel(X'*X_sampled,ker_params);
    ker_params.X = X_sampled;
    W = calc_kernel(X_sampled'*X_sampled,ker_params);
    if strcmp('eig','svd')
        [U,S,V] = svds(W,k);
        W_pinv = V*pinv(S)*U';
    else
        if (k ~= dim)
            [V,D] = eigs(W,k,'la');
        else
            [V,D] = eig(W);
        end
        W_pinv = V*pinv(D)*V';
    end
    K_coreset = C*W_pinv*C';
    norm_coreset(i) = norm(K - K_coreset,'fro')/norm(K,'fro');
    disp(['coreset, c/N = ',num2str(percent(i)),...
        '% err = ',num2str(norm_coreset(i))]);
end
disp('');

%% SVD approximation
norm_svd = zeros(length(percent),1);
for i = 1:length(c)
    [U,S,V] = svds(K,c(i));
    K_svd = U*S*V';
    norm_svd(i) = norm(K - K_svd,'fro')/norm(K,'fro');
    disp(['SVD, c/N = ',num2str(percent(i)),...
        '% err = ',num2str(norm_svd(i))]);
end
disp('');

%% save results

% save('RESULTS_FIGURE_1a','percent','norm_uni','norm_colnorm',...
%     'norm_diag','norm_kmeans','norm_coreset','norm_svd');

%% show graphs
% load RESULTS_FIGURE_1a

%%
load mycolormap
figure
hold all;
plot(percent,norm_coreset,'-o','Color',my_colors(3,:),'LineWidth',1.5,'MarkerFaceColor',my_colors(3,:));
plot(percent,norm_kmeans,'-s','Color',my_colors(2,:),'LineWidth',1.5,'MarkerFaceColor',my_colors(2,:));
plot(percent,norm_uni,'-^','Color',my_colors(1,:),'LineWidth',1.5,'MarkerFaceColor',my_colors(1,:)); 
plot(percent,norm_diag,'-d','Color',my_colors(4,:),'LineWidth',1.5,'MarkerFaceColor',my_colors(4,:));
plot(percent,norm_colnorm,'-p','Color',my_colors(5,:),'LineWidth',1.5,'MarkerFaceColor',my_colors(5,:));
plot(percent,norm_svd,'->','Color',my_colors(7,:),'LineWidth',1.5,'MarkerFaceColor',my_colors(7,:));
h=legend('Coreset','Kmeans','Uniform','Diag','Col-norm','SVD');
set(h,'FontName','Times New Roman');
% title('Approximation Error vs. Sampling Ratio (c/N) - USPS Polynomial 4','FontSize',12);
xlabel('(c/N) ratio','FontSize',12,'FontName','Times New Roman');
ylabel('Normalized Approximation Error','FontSize',12,'FontName','Times New Roman');
hold off
box on