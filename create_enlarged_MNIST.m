% ========================================================================
% Author: Alona Golts (zadneprovski@gmail.com)
% Date: 05-04-2016
%
% This script creates an enlarged version of a database. In this case
% MNIST, but not necessarily. The enlarged dataset contains the original
% images with single pixel shift in all directions (left, right, up, down
% and all 4 diagonal combinations). In case of MNIST, it divides the
% dataset into 9 parts and saves it as 9 mini-batches in disk. The new
% datasets will be saved in the '...\databases' folder under the name:
% MNIST_EX9_i, when i=1..9. 
% ========================================================================

clear all
close all

load MNIST
img = train_img;
dim = 28; % dimension of MNIST digits in both height and width
num_batches = 9; % desired number of mini-batches
num_examples = 60000; % original number of train examples in MNIST

% shift to the right direction (left column is zeros)
img_right = zeros(size(img));
img_right(dim+1:end,:) = img(1:end-dim,:);
img_right(1:dim,:) = 0;

% shift to the left direction (right column is zeros)
img_left = zeros(size(img));
img_left(1:end-dim,:) = img(dim+1:end,:);
img_left(end-dim+1:end) = 0;

% shift up (lower row is zeros)
img_up = zeros(size(img));
img_up(1:end-1,:) = img(2:end,:);
img_up(dim:dim:end,:) = 0;

% shift down (upper row is zeros)
img_down = zeros(size(img));
img_down(2:end,:) = img(1:end-1,:);
img_down(1:dim:end,:) = 0;

% shift right-up
img_right_up = zeros(size(img));
img_right_up(dim+1:end,:) = img(1:end-dim,:);
img_right_up(1:dim,:) = 0;
img_right_up(1:end-1,:) = img_right_up(2:end,:);
img_right_up(dim:dim:end,:) = 0;

% shift right-down
img_right_down = zeros(size(img));
img_right_down(dim+1:end,:) = img(1:end-dim,:);
img_right_down(1:dim,:) = 0;
img_right_down(2:end,:) = img_right_down(1:end-1,:);
img_right_down(1:dim:end,:) = 0;

% shift left-up
img_left_up = zeros(size(img));
img_left_up(1:end-dim,:) = img(dim+1:end,:);
img_left_up(end-dim+1:end) = 0;
img_left_up(1:end-1,:) = img_left_up(2:end,:);
img_left_up(dim:dim:end,:) = 0;

% shift left-down
img_left_down = zeros(size(img));
img_left_down(1:end-dim,:) = img(dim+1:end,:);
img_left_down(end-dim+1:end) = 0;
img_left_down(2:end,:) = img_left_down(1:end-1,:);
img_left_down(1:dim:end,:) = 0;

% combine all of the shifted digits into one set
extended_train = [img, img_right, img_left, img_up, img_down, ...
    img_right_up, img_right_down, img_left_up, img_left_down];
extended_train_lbl = [train_lbl, train_lbl, train_lbl, train_lbl, train_lbl,...
    train_lbl, train_lbl, train_lbl, train_lbl];

% scramble the digits randomly and create an enlarged dataset
indices = randperm(num_batches*num_examples);
extended_train = extended_train(:,indices);
extended_train_lbl = extended_train_lbl(:,indices);


% save mini-batches of data separately on disk
path = [pwd,'\databases\MNIST_EX',num2str(num_batches),'_'];

for i = 1:num_batches
    train_img = extended_train(:,(num_examples*(i-1)+1):num_examples*i);
    train_lbl = extended_train_lbl(:,(num_examples*(i-1)+1):num_examples*i);
    save([path,num2str(i)],'train_img','train_lbl');
end

% train_img = extended_train(:,1:60000);
% train_lbl = extended_train_lbl(:,1:60000);
% save([path,'\MNIST_EX9_1'],'train_img','train_lbl');
% 
% train_img = extended_train(:,60001:120000);
% train_lbl = extended_train_lbl(:,60001:120000);
% save([path,'\MNIST_EX9_2'],'train_img','train_lbl');
% 
% train_img = extended_train(:,120001:180000);
% train_lbl = extended_train_lbl(:,120001:180000);
% save([path,'\MNIST_EX9_3'],'train_img','train_lbl');
% 
% train_img = extended_train(:,180001:240000);
% train_lbl = extended_train_lbl(:,180001:240000);
% save([path,'\MNIST_EX9_4'],'train_img','train_lbl');
% 
% train_img = extended_train(:,240001:300000);
% train_lbl = extended_train_lbl(:,240001:300000);
% save([path,'\MNIST_EX9_5'],'train_img','train_lbl');
% 
% train_img = extended_train(:,300001:360000);
% train_lbl = extended_train_lbl(:,300001:360000);
% save([path,'\MNIST_EX9_6'],'train_img','train_lbl');
% 
% train_img = extended_train(:,360001:420000);
% train_lbl = extended_train_lbl(:,360001:420000);
% save([path,'\MNIST_EX9_7'],'train_img','train_lbl');
% 
% train_img = extended_train(:,420001:480000);
% train_lbl = extended_train_lbl(:,420001:480000);
% save([path,'\MNIST_EX9_8'],'train_img','train_lbl');
% 
% train_img = extended_train(:,480001:540000);
% train_lbl = extended_train_lbl(:,480001:540000);
% save([path,'\MNIST_EX9_9'],'train_img','train_lbl');
