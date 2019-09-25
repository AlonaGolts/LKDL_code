clear all
close all

load randomfaces4ar

num_classes = 100;
faces_per_class = 26;
train_per_class = 20;
test_per_class = faces_per_class - train_per_class;
training_feats = [];
testing_feats = [];
H_train = [];
H_test = [];
train_filenames = cell(1,num_classes);
test_filenames = cell(1,num_classes);

for i = 1:num_classes
    ind = randperm(faces_per_class);
    find_vec = find(labelMat(i,:)==1);
    training_feats = [training_feats featureMat(:,find_vec(ind(1:train_per_class)))];
    testing_feats = [testing_feats featureMat(:,find_vec(ind(train_per_class+1:end)))];
    H_train = [H_train labelMat(:,find_vec(ind(1:train_per_class)))];
    H_test = [H_test labelMat(:,find_vec(ind(train_per_class+1:end)))];
    train_filenames{i} = cell(1,train_per_class);
    for j = 1:train_per_class
        train_filenames{i}{j} = filenameMat{i}{ind(j)};
    end
    for j = 1:test_per_class
        test_filenames{i}{j} = filenameMat{i}{ind(train_per_class+j)};
    end
end

save('AR','training_feats','testing_feats','H_train','H_test',...
    'train_filenames','test_filenames');