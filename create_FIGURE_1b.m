% ========================================================================
% create figure 1(b) - classification accuracy as a function of c/N- percent of samples in Nystrom
% if you want to load the existing results, skip all the way to "show graphs"
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
% ========================================================================

%% load database
load USPS
train_size = size(train_lbl,2);

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

%% Check classification accuracy as a function of c/N

goal = 'check c';
range = round((0.05:0.05:0.5)*train_size);
%% LKDL with coreset sampling
classify_params.alg_type = 'KSVD';
classify_params.linear_approx = 1;
classify_params.samp_method = 'coreset';
results = classify_main(classify_params,goal,range);
LKDL_coreset = results.results_mat(2,:);
%% LKDL with uniform sampling
classify_params.samp_method = 'uniform';
results = classify_main(classify_params,goal,range);
LKDL_uniform = results.results_mat(2,:);
%% LKDL with kmeans 
classify_params.samp_method = 'kmeans';
results = classify_main(classify_params,goal,range);
LKDL_kmeans = results.results_mat(2,:);
%% LKDL with diagonal sampling
classify_params.samp_method = 'diag';
results = classify_main(classify_params,goal,range);
LKDL_diag = results.results_mat(2,:);
%% LKDL with column norm sampling
classify_params.samp_method = 'col_norm';
results = classify_main(classify_params,goal,range);
LKDL_colnorm = results.results_mat(2,:);
%% constant run of KKSVD (noise sigma = 0)
goal = 'check noise sigma';
classify_params.alg_type = 'KKSVD';
classify_params.linear_approx = 0;
results = classify_main(classify_params,goal,0);
KKSVD = results.results_mat(2,:)*ones(1,length(range));
%% constant run of KSVD (noise sigma = 0)
classify_params.alg_type = 'KSVD';
classify_params.card = 3;
results = classify_main(classify_params,goal,0);
KSVD = results.results_mat(2,:)*ones(1,length(range));

%% save results
% save('RESULTS_FIGURE_1b','LKDL_coreset','LKDL_kmeans','LKDL_uniform','LKDL_diag',...
%     'LKDL_colnorm','KSVD','KKSVD');

%% show graphs
% load RESULTS_FIGURE_1b

%% 
load mycolormap
train_size = 7291;
c = 0.05:0.05:0.5;
figure
hold all;
plot(c,LKDL_coreset,'-o','Color',my_colors(3,:),'MarkerFaceColor',my_colors(3,:),'LineWidth',1.5);
plot(c,LKDL_kmeans,'-s','Color',my_colors(2,:),'MarkerFaceColor',my_colors(2,:),'LineWidth',1.5); 
plot(c,LKDL_uniform,'-^','Color',my_colors(1,:),'MarkerFaceColor',my_colors(1,:),'LineWidth',1.5); 
plot(c,LKDL_diag,'-d','Color',my_colors(4,:),'MarkerFaceColor',my_colors(4,:),'LineWidth',1.5);
plot(c,LKDL_colnorm,'-p','Color',my_colors(5,:),'MarkerFaceColor',my_colors(5,:),'LineWidth',1.5);
plot(c,KSVD,'--','Color',my_colors(6,:),'LineWidth',3); 
plot(c,KKSVD,'-.','Color',my_colors(8,:),'LineWidth',3);
h=legend('LKDL Coreset','LKDL Kmeans','LKDL Uniform','LKDL Diag','LKDL Col-norm','KSVD','KKSVD','Location','NorthEast');
set(h,'FontName','Times New Roman');
xlabel('(c/N) ratio','FontSize',12,'FontName','Times New Roman');
ylabel('Classification Accuracy','FontSize',12,'FontName','Times New Roman');
% title('Classifiaction Accuracy vs. Sampling Ratio (c/N) - USPS Polynomial 2');
ylim([0.953 0.97]);
box on