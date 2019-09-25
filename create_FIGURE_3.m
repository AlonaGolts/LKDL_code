% ========================================================================
% create figure 3 - classification accuracy and runtime as a function of size of train set
% if you want to load the existing results, skip all the way to "show graphs"
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
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
classify_params.num_runs = 3;
classify_params.train_per_class = 0;
classify_params.test_per_class = 0;
classify_params.num_atoms = 700;
classify_params.iter = 2;
classify_params.card = 11;
classify_params.ker_type = 'Polynomial';
classify_params.ker_param_1 = 2;
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

%% KSVD
% goal = 'check num train';
% range = 10000:10000:60000;
% results  = classify_main(classify_params,goal,range);
% KSVD = results.results_mat;
%% KKSVD
% classify_params.alg_type = 'KKSVD';
% results  = classify_main(classify_params,goal,range);
% KKSVD = results.results_mat;
%% LKDL coreset
classify_params.alg_type = 'KSVD';
classify_params.linear_approx = 1;
classify_params.samp_method = 'coreset';
results  = classify_main(classify_params,goal,range);
LKDL_coreset = results.results_mat;

%% LKDL kmeans
classify_params.samp_method = 'kmeans';
results  = classify_main(classify_params,goal,range);
LKDL_kmeans = results.results_mat;

%% save results
% save('RESULTS_FIGURE_3','range','KSVD','KKSVD','LKDL_coreset','LKDL_kmeans');

%% show graphs
% load RESULTS_FIGURE_3

%% accuracy fig
load mycolormap
figure; 
hold on;
plot(range,KSVD(2,:),'-o','Color',my_colors(4,:),'MarkerFaceColor',my_colors(4,:),'LineWidth',1.5); 
plot(range,KKSVD(2,:),'-s','Color',my_colors(5,:),'MarkerFaceColor',my_colors(5,:),'LineWidth',1.5);
plot(range,LKDL_coreset(2,:),'-^','Color',my_colors(6,:),'MarkerFaceColor',my_colors(6,:),'LineWidth',1.5);
hold off;
ylim([0.97 0.99]);
xlabel('# Training Samples','FontSize',12,'FontName','Times New Roman');
ylabel('Classification Accuracy','FontSize',12,'FontName','Times New Roman');
h=legend('KSVD','KKSVD','LKDL Coreset','Location','best');
set(h,'FontName','Times New Roman');
% title('Classifcation Accuracy vs. # Training Samples');
set(gca,'XTick',range);
set(gca,'XTickLabel',{'10,000','20,000','30,000','40,000','50,000','60,000'});
box on

%% training time fig
figure; 
hold on
plot(range,KSVD(6,:),'-o','Color',my_colors(4,:),'MarkerFaceColor',my_colors(4,:),'LineWidth',1.5); 
plot(range,KKSVD(6,:),'-s','Color',my_colors(5,:),'MarkerFaceColor',my_colors(5,:),'LineWidth',1.5); 
plot(range,LKDL_coreset(4,:)+ LKDL_coreset(6,:),'-^','Color',my_colors(6,:),'MarkerFaceColor',my_colors(6,:),'LineWidth',1.5);
hold off
xlabel('# Training Samples','FontSize',12,'FontName','Times New Roman');
ylabel('Training Time log[sec]','FontSize',12,'FontName','Times New Roman');
h=legend('KSVD','KKSVD','LKDL Coreset','Location','best');
set(h,'FontName','Times New Roman');
% title('Training Time vs. # Training Samples');
set(gca,'YScale','log');
set(gca,'XTick',range);
set(gca,'XTickLabel',{'10,000','20,000','30,000','40,000','50,000','60,000'});
box on

%% test time fig
figure; 
hold on
plot(range,KSVD(7,:),'-o','Color',my_colors(4,:),'MarkerFaceColor',my_colors(4,:),'LineWidth',1.5); 
plot(range,KKSVD(7,:),'-s','Color',my_colors(5,:),'MarkerFaceColor',my_colors(5,:),'LineWidth',1.5); 
plot(range,LKDL_coreset(5,:)+ LKDL_coreset(7,:),'-^','Color',my_colors(6,:),'MarkerFaceColor',my_colors(6,:),'LineWidth',1.5);
hold off
xlabel('# Training Samples','FontSize',12,'FontName','Times New Roman');
ylabel('Test Time log[sec]','FontSize',12,'FontName','Times New Roman');
h=legend('KSVD','KKSVD','LKDL Coreset','Location','best');
set(h,'FontName','Times New Roman');
% title('Training Time vs. # Training Samples');
set(gca,'YScale','log');
set(gca,'XTick',range);
set(gca,'XTickLabel',{'10,000','20,000','30,000','40,000','50,000','60,000'});
box on