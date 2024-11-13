%% Bees + LPQ Image Feature Selection

%% Making Things Ready !!!
clc;
clear; 
warning('off');

%% LPQ Feature Extraction
% Read input images
path='exp';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);
end;
% Color to Gray Conversion
% for i = 1 : filesnumber(1,1)
% gray{i}=rgb2gray(images{i});
% disp(['To Gray :   ' num2str(i) ]);end;
% Contrast Adjustment
for i = 1 : filesnumber(1,1)
adjusted2{i}=imadjust(images{i});
disp(['Contrast Adjust :   ' num2str(i) ]);end;
% Resize Image
for i = 1 : filesnumber(1,1)
resized2{i}=imresize(adjusted2{i}, [256 256]);
disp(['Image Resized :   ' num2str(i) ]);end;

%% LPQ Features
clear LPQ_tmp;clear LPQ_Features;

winsize=9;

for i = 1 : filesnumber(1,1)
LPQ_tmp{i}=lpq(resized2{i},winsize);
disp(['Extract LPQ :   ' num2str(i) ]);end;
for i = 1 : filesnumber(1,1)
LPQ_Features(i,:)=LPQ_tmp{i};end;

%% Labeling for Classification
sizefinal=size(LPQ_Features);
sizefinal=sizefinal(1,2);
%------------------------------------
LPQ_Features(1:100,sizefinal+1)=1;
LPQ_Features(101:200,sizefinal+1)=2;
LPQ_Features(201:300,sizefinal+1)=3;

%% Bees Feature Selection
% Data Preparation
x=LPQ_Features(:,1:end-1)';
t=LPQ_Features(:,end)';
data.x=x;
data.t=t;
data.nx=size(x,1);
data.nt=size(t,1);
data.nSample=size(x,2);

%% Number of Desired BEE Features
%----------------------------------
nf=150;
%----------------------------------

%% Cost Function
CostFunction=@(u) FeatureSelectionCost(u,nf,data);
% Number of Decision Variables
nVar=data.nx;
% Size of Decision Variables Matrix
VarSize=[1 nVar];
% Lower Bound of Variables
VarMin=0;
% Upper Bound of Variables
VarMax=1;

%% Bees Algorithm Parameters
MaxIt = 50;                                % Maximum Number of Iterations
nScoutBee = 2;                            % Number of Scout Bees
nSelectedSite = round(0.5*nScoutBee);     % Number of Selected Sites
nEliteSite = round(0.4*nSelectedSite);    % Number of Selected Elite Sites
nSelectedSiteBee = round(0.5*nScoutBee);  % Number of Recruited Bees for Selected Sites
nEliteSiteBee = 2*nSelectedSiteBee;       % Number of Recruited Bees for Elite Sites
r = 0.1*(VarMax-VarMin);	              % Neighborhood Radius
rdamp = 0.96;                             % Neighborhood Radius Damp Rate

%% Basics
% Empty Bee Structure
empty_bee.Position = [];
empty_bee.Cost = [];
empty_bee.Out = [];

% Bees Array
bee = repmat(empty_bee, nScoutBee, 1);
% Create New Solutions
for i = 1:nScoutBee
bee(i).Position = unifrnd(VarMin, VarMax, VarSize);
[bee(i).Cost bee(i).Out] = CostFunction(bee(i).Position);end;
% Sort
[~, SortOrder] = sort([bee.Cost]);
bee = bee(SortOrder);
% Update Best Solution Ever Found
BestSol = bee(1);
% Array to Hold Best Cost Values
BestCost = zeros(MaxIt, 1);

%-------------------------------------------------
%% Bees Algorithm Main Loop
for it = 1:MaxIt
% Elite Sites
for i = 1:nEliteSite
bestnewbee.Cost = inf;
for j = 1:nEliteSiteBee
newbee.Position = PerformBeeDance(bee(i).Position, r);
[newbee.Cost newbee.Out] = CostFunction(newbee.Position);
if newbee.Cost<bestnewbee.Cost
bestnewbee = newbee;end;end;
if bestnewbee.Cost<bee(i).Cost
bee(i) = bestnewbee;end;end;
% Selected Non-Elite Sites
for i = nEliteSite+1:nSelectedSite
bestnewbee.Cost = inf;
for j = 1:nSelectedSiteBee
newbee.Position = PerformBeeDance(bee(i).Position, r);
[newbee.Cost newbee.Out] = CostFunction(newbee.Position);
if newbee.Cost<bestnewbee.Cost
bestnewbee = newbee;
end;end;
if bestnewbee.Cost<bee(i).Cost
bee(i) = bestnewbee;
end;end;
% Non-Selected Sites
for i = nSelectedSite+1:nScoutBee
bee(i).Position = unifrnd(VarMin, VarMax, VarSize);
[bee(i).Cost bee(i).Out] = CostFunction(bee(i).Position);end;
% Sort
[~, SortOrder] = sort([bee.Cost]);
bee = bee(SortOrder);
% Update Best Solution Ever Found
BestSol = bee(1);
% Store Best Cost Ever Found
BestCost(it) = BestSol.Cost;
% Display Iteration Information
disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
% Damp Neighborhood Radius
r = r*rdamp;end;

% Plot ---------------------------------------------- 
plot(BestCost, '--k','linewidth',2);
xlabel('Iteration');
ylabel('Bees Cost');

%---------------------------------------------------
%% Creating Bees + LPQ Features Matrix
% Extracting Data
RealData=data.x';
% Extracting Labels
RealLbl=data.t';
FinalFeaturesInd=BestSol.Out.S;
% Sort Features
FFI=sort(FinalFeaturesInd);
% Select Final Features
BEE_LPQ_Features=RealData(:,FFI);
% Adding Labels
BEE_LPQ_Features_Lbl=BEE_LPQ_Features;
BEE_LPQ_Features_Lbl(:,end+1)=RealLbl;
LPQ_Bees=BEE_LPQ_Features_Lbl;


%% Classification
% KNN------------------------------------
% Split data into features and labels
X = LPQ_Bees(:, 1:end-1);  % Features
y = LPQ_Bees(:, end);      % Labels
% Divide the data into training and testing sets
cv = cvpartition(y, 'HoldOut', 0.3);  % 70% training, 30% testing
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);
% Train a KNN classifier
k = 3;  % You can adjust the number of neighbors
knn_model = fitcknn(X_train, y_train, 'NumNeighbors', k);
% Test the classifier
y_pred = predict(knn_model, X_test);
% Calculate test accuracy
test_accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('KNN Test Accuracy: %.2f%%\n', test_accuracy);

%--------------------------------------------------
% Shallow NN------------------------------------
% Split data into features and labels
X = LPQ_Bees(:, 1:end-1)';  % Features (transpose for neural network input)
y = LPQ_Bees(:, end)';      % Labels (transpose for neural network target)
% Convert labels to categorical format for neural network
num_classes = numel(unique(y));
y_onehot = full(ind2vec(y));  % Convert numeric labels to one-hot encoding
% Divide the data into training and testing sets
cv = cvpartition(y, 'HoldOut', 0.3);  % 70% training, 30% testing
trainInd = training(cv);
testInd = test(cv);
X_train = X(:, trainInd);
y_train = y_onehot(:, trainInd);
X_test = X(:, testInd);
y_test = y(testInd);

% Define and configure a shallow neural network
hiddenLayerSize = 10;  % Number of neurons in the hidden layer, can be adjusted
net = feedforwardnet(hiddenLayerSize);
% Set up training parameters (optional)
net.trainParam.epochs = 100;     % Number of training epochs
net.trainParam.showWindow = false;  % Disable training GUI
% Train the network
net = train(net, X_train, y_train);
% Test the network
y_pred_onehot = net(X_test);
[~, y_pred] = max(y_pred_onehot, [], 1);  % Convert one-hot predictions to class labels
% Calculate test accuracy
test_accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('Shallow NN Test Accuracy: %.2f%%\n', test_accuracy);

% Ensemble Subspace KNN------------------------------------
% Split data into features and labels
X = LPQ_Bees(:, 1:end-1);  % Features
y = LPQ_Bees(:, end);      % Labels
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

