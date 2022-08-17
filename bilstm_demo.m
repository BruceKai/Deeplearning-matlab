clc,clear;

%% Load the training data
% trainX: an array with shape of (n, c, t). n represents the number
%            of training samples, c is the number of features, t is the
%             length of time sequence.
% trainY: an array with shape of (n,). n represents the number of training
%            samples. 
rootDir = 'Deeplearning-matlab';
trainingData = importdata(fullfile(rootDir,'train.mat'));
trainX = trainingData.trainx;
trainY = trainingData.trainy+1;
Xtrain = cell({});
for i = 1:size(trainX,1)
    Xtrain{i,1} = squeeze(trainX(i,:,:));
end
Ytrain = categorical(trainY');

%% Load the validation data
valData = importdata(fullfile(rootDir,'test.mat'));
valX = valData.testx;
valY = valData.testy+1;
Yval = categorical(valY');
Xval = cell({});
for i = 1:size(valX,1)
    Xval{i,1} = squeeze(valX(i,:,:));
end
valDataSet = cell({Xval,Yval});

%% Create bilstm model
% numFeatures: The number of expected features in input data
% numHiddens: The number of features in the hidden state
% numClasses: The number of classess
numFeatures = 8;
numHiddens = 256;
numClasses = 5;

netLayers = [
    sequenceInputLayer(numFeatures,"Name","input")
    bilstmLayer(numHiddens,"Name","bilstm_1",'OutputMode','last')
    bilstmLayer(numHiddens,"Name","bilstm_2",'OutputMode','last')
    dropoutLayer(0.5,"Name","dropout")
    flattenLayer("Name","flatten")
    fullyConnectedLayer(numClasses,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classification")];

%% Set the hyper parameters for unet training 
options = trainingOptions('adam', ...                          
                                        'InitialLearnRate',1e-4, ...    
                                        'Plots','training-progress',...
                                        'MaxEpochs',60, ... 
                                        'MiniBatchSize',128,...
                                        'VerboseFrequency',1,...
                                        'ExecutionEnvironment', 'auto',...
                                        'Shuffle','every-epoch',...
                                         'ValidationData',valDataSet,...
                                         'ValidationFrequency',1,...
                                         'WorkerLoad',4,...
                                         'CheckPointPath',rootDir);
% start training
net = trainNetwork(Xtrain,Ytrain, netLayers, options);

%% Save and load model
save('bilstm.mat','net');
bilstm = importdata('bilstm.mat');

%% Accuracy assessment
pred = classify(bilstm, Xval);
[confusionMatrix,order] = confusionmat(categorical(valY),pred);
cm = confusionchart(confusionMatrix);

% caculate user accuracy and mapping accuracy
confusionMatrix = [confusionMatrix, zeros(size(confusionMatrix,1),1)];
confusionMatrix = [confusionMatrix; zeros(1,size(confusionMatrix,2))];
confusionMatrix(1:end-1,end) = confusionMatrix(sub2ind(size(confusionMatrix),1:numClasses,1:numClasses))...
                                                ./sum(confusionMatrix(1:end-1,1:end-1),2)';
confusionMatrix(end,1:end-1) = confusionMatrix(sub2ind(size(confusionMatrix),1:numClasses,1:numClasses))...
                                                ./sum(confusionMatrix(1:end-1,1:end-1),1);    
confusionMatrix(end,end) = sum(confusionMatrix(sub2ind(size(confusionMatrix),1:numClasses,1:numClasses)))...
                                                ./sum(sum(confusionMatrix(1:end-1,1:end-1)));
                                            
mappingAccuracy = confusionMatrix(end,1:end-1);
userAccuracy = confusionMatrix(1:end-1,end);
totalAccuracy = confusionMatrix(end,end);
