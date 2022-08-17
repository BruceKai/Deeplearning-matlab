clc,clear;
%% Load training datasets
% rootDir: the root direction of the training data
rootDir = 'data';
imageDir = fullfile(rootDir,'img');
labelDir = fullfile(rootDir,'lbl');
classNames = ["background","foreground"];
labelIDs = [0,255];
imageDataSet = imageDatastore(imageDir);
labelDataSet = pixelLabelDatastore(labelDir,classNames,labelIDs);

% create the training dataset
trainingDataSet = pixelLabelImageDatastore(imageDataSet,labelDataSet);

%% Load validation datasets
rootDir = 'data';
imageDir = fullfile(rootDir,'img');
labelDir = fullfile(rootDir,'lbl');
classNames = ["background","foreground"];
labelIDs = [0,255];
imageDataSet = imageDatastore(imageDir);
labelDataSet = pixelLabelDatastore(labelDir,classNames,labelIDs);

% create the training dataset
valDataSet = pixelLabelImageDatastore(imageDataSet,labelDataSet);

%% Create a unet model for segmentation
% imageSize: Input training image size (height, width, channels)
% numClasses: Number of classes
% encoderDepth: Encoder depth
imageSize = [256,256,4]; 
numClasses = 2;
encoderDepth = 4;
[netLayers,outsize] = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth);

% Applying dice loss to address class imbalance.
netLayers = replaceLayer(netLayers,'Segmentation-Layer', dicePixelClassificationLayer('Name','Segmentation-Layer'));

%% Set the hyper parameters for unet training 
options = trainingOptions('adam', ...                          
                                        'InitialLearnRate',1e-3, ...    
                                        'Plots','training-progress',...
                                        'MaxEpochs',10, ... 
                                        'MiniBatchSize',32,...
                                        'VerboseFrequency',20,...
                                        'ExecutionEnvironment', 'multi-gpu',...
                                        'Shuffle','every-epoch',...
                                         'ValidationData',valDataSet,...
                                         'ValidationFrequency',50,...
                                         'WorkerLoad',4,...
                                         'CheckPointPath',rootDir);
% start training
net = trainNetwork(trainingDataSet, netLayers, options);

%% Save and load model
save('unet_test.mat','net');
unet = importdata('unet_test.mat');
