% ========================================================================
% create figure 2 - classification accuracy in the presence of noise & missing pixels
% if you want to load the existing results, skip all the way to "show graphs"
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
% ========================================================================

%% load database
load USPS
train_size = size(train_img,2);

%% define all paramters - obligatory step
classify_params.run_mode = 'regular';
classify_params.pre_process = 'mean_std';
classify_params.train_images = train_img;
classify_params.train_labels = train_lbl;
classify_params.test_images = test_img;
classify_params.test_labels = test_lbl;
classify_params.num_classes = 10;
classify_params.alg_type = 'KSVD';
classify_params.num_runs = 10;
classify_params.train_per_class = 0;
classify_params.test_per_class = 0;
classify_params.num_atoms = 300;
classify_params.iter = 5;
classify_params.card = 5;
classify_params.ker_type = 'Polynomial';
classify_params.ker_param_1 = 2;
classify_params.ker_param_2 = 0;
classify_params.linear_approx = 1;
classify_params.samp_method = 'coreset';
classify_params.c = round(0.2*(train_size));
classify_params.k = 256;
classify_params.train_size = train_size;
classify_params.sigma = 0;
classify_params.missing_pixels = 0;
classify_params.init_dic = 'partial';
classify_params.num_batches = 1;

%% create results for figure 2(a) - Accuracy as a function of Gaussian noise corruption level
goal = 'check noise sigma';
range = [0:0.2:2];
%% LKDL coreset
classify_params.alg_type = 'KSVD';
classify_params.linear_approx = 1;
classify_params.samp_method = 'coreset';
results = classify_main(classify_params,goal,range);
LKDL_noise_coreset = results.results_mat(2,:);
%% LKDL uniform
classify_params.samp_method = 'uniform';
results = classify_main(classify_params,goal,range);
LKDL_noise_uniform = results.results_mat(2,:);
%% LKDL kmeans
classify_params.samp_method = 'kmeans';
results = classify_main(classify_params,goal,range);
LKDL_noise_kmeans = results.results_mat(2,:);
%% LKDL diag
classify_params.samp_method = 'diag';
results = classify_main(classify_params,goal,range);
LKDL_noise_diag = results.results_mat(2,:);
%% LKDL column-norm
classify_params.samp_method = 'col_norm';
results = classify_main(classify_params,goal,range);
LKDL_noise_colnorm = results.results_mat(2,:);
%% KKSVD
classify_params.linear_approx = 0;
classify_params.alg_type = 'KKSVD';
results = classify_main(classify_params,goal,range);
KKSVD_noise = results.results_mat(2,:);
%% KSVD
classify_params.alg_type = 'KSVD';
classify_params.card = 3;   %% different cardinality for KSVD (performs better)
results = classify_main(classify_params,goal,range);
KSVD_noise = results.results_mat(2,:);

%% create results for figure 2(b) - Accuracy as a function of percent of missing pixels
goal = 'check missing pixels';
range = 0:0.1:0.9;
%% LKDL coreset
classify_params.alg_type = 'KSVD';
classify_params.linear_approx = 1;
classify_params.card = 5;
classify_params.samp_method = 'coreset';
results = classify_main(classify_params,goal,range);
LKDL_pixels_coreset = results.results_mat(2,:);
%% LKDL uniform
classify_params.samp_method = 'uniform';
results = classify_main(classify_params,goal,range);
LKDL_pixels_uniform = results.results_mat(2,:);
%% LKDL kmeans
classify_params.samp_method = 'kmeans';
results = classify_main(classify_params,goal,range);
LKDL_pixels_kmeans = results.results_mat(2,:);
%% LKDL diag
classify_params.samp_method = 'diag';
results = classify_main(classify_params,goal,range);
LKDL_pixels_diag = results.results_mat(2,:);
%% LKDL column-norm
classify_params.samp_method = 'col_norm';
results = classify_main(classify_params,goal,range);
LKDL_pixels_colnorm = results.results_mat(2,:);
%% KKSVD
classify_params.linear_approx = 0;
classify_params.alg_type = 'KKSVD';
results = classify_main(classify_params,goal,range);
KKSVD_pixels = results.results_mat(2,:);
%% KSVD
classify_params.alg_type = 'KSVD';
classify_params.card = 3;
results = classify_main(classify_params,goal,range);
KSVD_pixels = results.results_mat(2,:);

%% save results
% save('RESULTS_FIGURE_2','LKDL_noise_coreset','LKDL_noise_uniform','LKDL_noise_kmeans',...
%     'LKDL_noise_diag','LKDL_noise_colnorm','KKSVD_noise','KSVD_noise',...
%     'LKDL_pixels_coreset','LKDL_pixels_uniform','LKDL_pixels_kmeans','LKDL_pixels_diag',...
%     'LKDL_pixels_colnorm','KKSVD_pixels','KSVD_pixels','classify_params');

%% show graphs
% load RESULTS_FIGURE_2

%%
load mycolormap
noise = 0:0.2:2;
pixels = 0:0.1:0.9;
figure; 
plot(noise,KSVD_noise,'-o','Color',my_colors(4,:),'MarkerFaceColor',my_colors(4,:),'LineWidth',1.5); hold on;
plot(noise,KKSVD_noise,'-s','Color',my_colors(5,:),'MarkerFaceColor',my_colors(5,:),'LineWidth',1.5); hold on;
plot(noise,LKDL_noise_coreset,'-^','Color',my_colors(6,:),'MarkerFaceColor',my_colors(6,:),'LineWidth',1.5); 
h=legend('KSVD','KKSVD','LKDL Coreset','Location','Best');
set(h,'FontName','Times New Roman');
xlabel('Noise Level','FontSize',12,'FontName','Times New Roman');
ylabel('Classification Accuracy','FontSize',12,'FontName','Times New Roman');
% title('Classifiaction Accuracy in the Presence of Noise');

figure; 
plot(pixels,KSVD_pixels,'-o','Color',my_colors(4,:),'MarkerFaceColor',my_colors(4,:),'LineWidth',1.5); hold on;
plot(pixels,KKSVD_pixels,'-s','Color',my_colors(5,:),'MarkerFaceColor',my_colors(5,:),'LineWidth',1.5); hold on;
plot(pixels,LKDL_pixels_coreset,'-^','Color',my_colors(6,:),'MarkerFaceColor',my_colors(6,:),'LineWidth',1.5); hold on;
h=legend('KSVD','KKSVD','LKDL Coreset','Location','Best');
set(h,'FontName','Times New Roman');
xlabel('% Missing Pixels','FontSize',12,'FontName','Times New Roman');
ylabel('Classification Accuracy','FontSize',12,'FontName','Times New Roman');
% title('Classifiaction Accuracy in the Presence of Missing Pixels');