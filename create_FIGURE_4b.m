% ========================================================================
% create figure 4(b) - classification accuracy as a function of k-
% dimension of signals in MNIST database. 
% if you want to load the existing results, skip all the way to "show graphs"
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 19-01-2016
% ========================================================================

%% load database
load MNIST
train_size = size(train_img,2);

%% define all paramters - obligatory step
classify_params.run_mode = 'regular';
classify_params.pre_process = 'std';
classify_params.train_images = train_img;
classify_params.train_labels = train_lbl;
classify_params.test_images = test_img;
classify_params.test_labels = test_lbl;
classify_params.num_classes = 10;
classify_params.alg_type = 'KSVD';
classify_params.num_runs = 20;
classify_params.train_per_class = 0;
classify_params.test_per_class = 0;
classify_params.num_atoms = 700;
classify_params.iter = 2;
classify_params.card = 11;
classify_params.ker_type = 'Gaussian';
classify_params.ker_param_1 = 1;
classify_params.ker_param_2 = 0;
classify_params.linear_approx = 0;
classify_params.samp_method = 'coreset';
classify_params.c = round(0.15*(train_size));
classify_params.k = 784;
classify_params.train_size = 60000;
classify_params.sigma = 0;
classify_params.missing_pixels = 0;
classify_params.init_dic = 'partial';
classify_params.num_batches = 1;

%% Check classification accuracy as a function of c=k - approximation dimension
goal = 'check approx dimension';
range = [64 128 256 512 784 1024 2048];

%% Fastfood features with KSVD
classify_params.linear_approx = 3;
results_Fastfood = classify_main(classify_params,goal,range);
Fastfood = results_Fastfood.results_mat(2,:);

%% Random Fourier Features with KSVD
classify_params.linear_approx = 2;
classify_params.alg_type = 'KSVD';
results_RFF = classify_main(classify_params,goal,range);
RFF = results_RFF.results_mat(2,:);

%% LKDL with kmeans constant c = 2048
classify_params.c = 2048;
results_LKDL_kmeans_const_c_2048 = classify_main(classify_params,'check k',range);
LKDL_kmeans_const_c_2048 = results_LKDL_kmeans_const_c_2048.results_mat(2,:);

%% constant run of KKSVD (noise sigma = 0)
goal = 'check noise sigma';
classify_params.alg_type = 'KKSVD';
classify_params.linear_approx = 0;
results_KKSVD = classify_main(classify_params,goal,0);
KKSVD = results_KKSVD.results_mat(2,:)*ones(1,length(range));

%% constant run of KSVD (noise sigma = 0)
classify_params.alg_type = 'KSVD';
classify_params.card = 3;
results_KSVD = classify_main(classify_params,goal,0);
KSVD = results_KSVD.results_mat(2,:)*ones(1,length(range));

%% save results
% save('RESULTS_FIGURE_4b','range','KSVD','KKSVD','LKDL_kmeans_const_c_2048','RFF','Fastfood');

%% show graphs
% load RESULTS_FIGURE_4b

%%
ind = 3:length(range);
load mycolormap
figure
hold all;
% plot(range(ind),LKDL_kmeans(ind),'-o','Color',my_colors(4,:),'MarkerFaceColor',my_colors(4,:),'LineWidth',1.5);
plot(range(ind),LKDL_kmeans_const_c_2048(ind),'-s','Color',my_colors(5,:),'MarkerFaceColor',my_colors(5,:),'LineWidth',1.5);
plot(range(ind),RFF(ind),'-^','Color',my_colors(6,:),'MarkerFaceColor',my_colors(6,:),'LineWidth',1.5);
plot(range(ind),Fastfood(ind),'-o','Color',my_colors(4,:),'MarkerFaceColor',my_colors(4,:),'LineWidth',1.5);
plot(range(ind),KSVD(ind),'--','Color',my_colors(3,:),'LineWidth',3); 
plot(range(ind),KKSVD(ind),'-.','Color',my_colors(8,:),'LineWidth',3);
h=legend('LKDL Kmeans','Random Fourier','Fastfood','KSVD','KKSVD','Location','Best');
set(h,'FontName','Times New Roman');
xlabel('k - approximation dimension','FontSize',12,'FontName','Times New Roman');
ylabel('Classification Accuracy','FontSize',12,'FontName','Times New Roman');
% title('MNIST');
ylim([0.97 0.985]);
xlim([min(range(ind))-10,max(range(ind))]);
set(gca,'XTick',range);
% set(gca,'XTickLabel',{'100','144','256','400','576','784','1024'});
box on