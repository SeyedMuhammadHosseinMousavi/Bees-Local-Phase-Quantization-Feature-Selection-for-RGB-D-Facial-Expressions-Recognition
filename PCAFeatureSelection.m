%% PCA + LPQ Image Feature Selection Pipeline

%% Making Things Ready !!!
clc;
clear; 
warning('off');

%% LPQ Feature Extraction
% Read input images
path = 'exp';
fileinfo = dir(fullfile(path, '*.jpg'));
filesnumber = size(fileinfo);
for i = 1 : filesnumber(1,1)
    images{i} = imread(fullfile(path, fileinfo(i).name));
    disp(['Loading image No :   ' num2str(i) ]);
end;

% Contrast Adjustment
for i = 1 : filesnumber(1,1)
    adjusted2{i} = imadjust(images{i});
    disp(['Contrast Adjust :   ' num2str(i) ]);
end;

% Resize Image
for i = 1 : filesnumber(1,1)
    resized2{i} = imresize(adjusted2{i}, [256 256]);
    disp(['Image Resized :   ' num2str(i) ]);
end;

%% LPQ Features
clear LPQ_tmp; clear LPQ_Features;
winsize = 9;
for i = 1 : filesnumber(1,1)
    LPQ_tmp{i} = lpq(resized2{i}, winsize);
    disp(['Extract LPQ :   ' num2str(i) ]);
end;

% Combine LPQ features into a single matrix
for i = 1 : filesnumber(1,1)
    LPQ_Features(i,:) = LPQ_tmp{i};
end;

%% Labeling for Classification
sizefinal = size(LPQ_Features);
sizefinal = sizefinal(1,2);
LPQ_Features(1:100, sizefinal+1) = 1;
LPQ_Features(101:200, sizefinal+1) = 2;
LPQ_Features(201:300, sizefinal+1) = 3;

%% PCA Feature Selection
% Data Preparation
x = LPQ_Features(:, 1:end-1);  % Features
t = LPQ_Features(:, end);      % Labels

% Apply PCA to reduce features
num_pca_components = 100;  % Number of principal components to retain
[coeff, score, ~] = pca(x);
pca_features = score(:, 1:num_pca_components);  % Selecting top components

% Combine the selected features with labels
LPQ_PCA_Features = [pca_features, t];

%% Classification

% KNN------------------------------------
% Split data into features and labels
X = LPQ_PCA_Features(:, 1:end-1);  % Features
y = LPQ_PCA_Features(:, end);      % Labels

% Divide the data into training and testing sets
cv = cvpartition(y, 'HoldOut', 0.3);  % 70% training, 30% testing
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Train a KNN classifier
k = 3;  % Number of neighbors
knn_model = fitcknn(X_train, y_train, 'NumNeighbors', k);

% Test the classifier
y_pred = predict(knn_model, X_test);

% Calculate test accuracy
test_accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('KNN Test Accuracy: %.2f%%\n', test_accuracy);

% Shallow Neural Network------------------------------------
% Transpose features for neural network input
X = LPQ_PCA_Features(:, 1:end-1)';  % Features
y = LPQ_PCA_Features(:, end)';      % Labels

% Convert labels to one-hot encoding
num_classes = numel(unique(y));
y_onehot = full(ind2vec(y));

% Divide the data into training and testing sets
cv = cvpartition(y, 'HoldOut', 0.3);
trainInd = training(cv);
testInd = test(cv);
X_train = X(:, trainInd);
y_train = y_onehot(:, trainInd);
X_test = X(:, testInd);
y_test = y(testInd);

% Define and configure a shallow neural network
hiddenLayerSize = 10;  % Number of neurons in hidden layer
net = feedforwardnet(hiddenLayerSize);
net.trainParam.epochs = 100;  % Training epochs
net.trainParam.showWindow = false;

% Train the network
net = train(net, X_train, y_train);

% Test the network
y_pred_onehot = net(X_test);
[~, y_pred] = max(y_pred_onehot, [], 1);  % Convert one-hot predictions to labels

% Calculate test accuracy
test_accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('Shallow NN Test Accuracy: %.2f%%\n', test_accuracy);

% Ensemble Subspace KNN------------------------------------
% Split data into features and labels
X = LPQ_PCA_Features(:, 1:end-1);  % Features
y = LPQ_PCA_Features(:, end);      % Labels

% Divide the data into training and testing sets
cv = cvpartition(y, 'HoldOut', 0.3);  % 70% training, 30% testing
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Train an ensemble classifier with subspace KNN
num_neighbors = 5;  % Number of neighbors for KNN
num_learners = 50;  % Number of learners in the ensemble
subspace_dim = ceil(size(X_train, 2) * 0.5);  % Subspace dimension, e.g., 50% of features

% Create and train the ensemble
ensemble_model = fitcensemble(X_train, y_train, ...
    'Method', 'Subspace', ...
    'NumLearningCycles', num_learners, ...
    'Learners', templateKNN('NumNeighbors', num_neighbors), ...
    'NPredToSample', subspace_dim);

% Test the classifier
y_pred = predict(ensemble_model, X_test);

% Calculate test accuracy
test_accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('Ensemble Subspace KNN Test Accuracy: %.2f%%\n', test_accuracy);

% Generate and display the confusion matrix
conf_matrix = confusionmat(y_test, y_pred);
disp('Confusion Matrix:');
disp(conf_matrix);

% Plot the confusion matrix as a heatmap
figure;
heatmap(unique(y), unique(y), conf_matrix, 'Colormap', parula, ...
    'ColorbarVisible', 'on');
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix for Ensemble Subspace KNN');
