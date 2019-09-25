% ========================================================================
% create figure 4(a) - classification accuracy as a function of k -
% dimension of features in USPS database
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
classify_params.num_runs = 20;
classify_params.train_per_class = 0;
classify_params.test_per_class = 0;
classify_params.num_atoms = 300;
classify_params.iter = 5;
classify_params.card = 5;
classify_params.ker_type = 'Gaussian';
classify_params.ker_param_1 = 1;
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

%% Check classification accuracy as a function of c=k - approximation dimension
goal = 'check approx dimension';
range = [64 128 256 512 784 1024];

%% Fastfood with KSVD
classify_params.linear_approx = 3;
Fastfood_results = classify_main(classify_params,goal,range);
Fastfood = Fastfood_results.results_mat(2,:);

%% Random Fourier Features (RFF) with KSVD
classify_params.linear_approx = 2;
RFF_results = classify_main(classify_params,goal,range);
RFF = RFF_results.results_mat(2,:);

%% LKDL with kmeans sampling, constant c = 20% = 1438 samples
classify_params.samp_method = 'kmeans';
classify_params.linear_approx = 1;
LKDL_kmeans_results_const_c = classify_main(classify_params,'check k',range);
LKDL_kmeans_const_c = LKDL_kmeans_results_const_c.results_mat(2,:);

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
% save('RESULTS_FIGURE_4a','range','KSVD','KKSVD','RFF','Fastfood','LKDL_kmeans_const_c');

%% show graphs
% load RESULTS_FIGURE_4a

%% Draw figure
ind = 1:length(range);
load mycolormap
figure
hold all;
plot(range(ind),LKDL_kmeans_const_c(ind),'-s','Color',my_colors(5,:),'MarkerFaceColor',my_colors(5,:),'LineWidth',1.5);
plot(range(ind),RFF(ind),'-^','Color',my_colors(6,:),'MarkerFaceColor',my_colors(6,:),'LineWidth',1.5);
plot(range(ind),Fastfood(ind),'-o','Color',my_colors(4,:),'MarkerFaceColor',my_colors(4,:),'LineWidth',1.5);
plot(range(ind),KSVD(ind),'--','Color',my_colors(3,:),'LineWidth',3); 
plot(range(ind),KKSVD(ind),'-.','Color',my_colors(8,:),'LineWidth',3);
h=legend('LKDL Kmeans','RFF','Fastfood','KSVD','KKSVD','Location','Best');
set(h,'FontName','Times New Roman');
xlabel('k - approximation dimension','FontSize',12,'FontName','Times New Roman');
ylabel('Classification Accuracy','FontSize',12,'FontName','Times New Roman');
ylim([0.935 0.963]);
xlim([min(range(ind)),max(range(ind))]);
set(gca,'XTick',range(ind));
box on