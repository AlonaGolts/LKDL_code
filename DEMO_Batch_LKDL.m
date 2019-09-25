% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% This demo demonstrates the mini-batch version of LKDL. It creates an
% enlarged version of MNIST using the function 'create_enlarged_MNIST', 
% then performs mini-batch RLS with and without LKDL. 
% ========================================================================

clear all
close all
clc

% create an enlarged train set of 540,000 examples
% create_enlarged_MNIST

load MNIST
train_size = size(train_img,2);

% define all paramters - obligatory step
classify_params.database = 'MNIST_EX';
classify_params.run_mode = 'regular';
classify_params.pre_process = 'mean_std';
classify_params.train_images = train_img;
classify_params.train_labels = train_lbl;
classify_params.test_images = test_img;
classify_params.test_labels = test_lbl;
classify_params.num_classes = 10;
classify_params.num_runs = 5;
classify_params.train_per_class = 0;
classify_params.test_per_class = 0;
classify_params.num_atoms = 700;
classify_params.iter = 2;
classify_params.card = 11;
classify_params.ker_type = 'Polynomial';
classify_params.ker_param_1 = 2;
classify_params.ker_param_2 = 0;
classify_params.linear_approx = 0;
classify_params.samp_method = 'kmeans';
classify_params.c = round(0.15*(train_size));
classify_params.k = 784;
classify_params.train_size = 60000;
classify_params.sigma = 0;
classify_params.missing_pixels = 0;
classify_params.init_dic = 'partial';

% define mini-batch parameters
classify_params.num_batches = 9;
classify_params.alg_type = 'RLS';
classify_params.minibatch = [100,20; 200,50];
classify_params.lam0 = 1;

% call mini-batch RLS without LKDL
results_KSVD = classify_main(classify_params,'check noise sigma',0);
results_KSVD = results_KSVD.results_mat;

% call mini-batch RLS with LKDL
classify_params.linear_approx = 1;
results_LKDL = classify_main(classify_params,'check noise sigma',0);
results_LKDL = results_LKDL.results_mat;
