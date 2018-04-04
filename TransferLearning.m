%===================================================================
% Intracerebral EEG artifact identification using convolutional neural networks
%===================================================================
%% Authors
%===================================================================
% Petr Nejedly*,1,2,3
% Jan Cimbalnik, 1
% Petr Klimes, 1, 2
% Filip Plesinger, 2
% Josef Halamek, 2,
% Vaclav Kremen, 3,6
% Benjamin Brinkmann, 3,6
% Martin Pail, 4
% Milan Brazdil, 4,5
% Gregory Worrell 3,6
% Pavel Jurak 2
%===================================================================
%% Affiliation
%===================================================================
% 1) International Clinical Research Center, St. Anne’s University Hospital, Brno, Czech Republic
% 2) Institute of Scientific Instruments, The Czech Academy of Sciences, Brno, Czech Republic
% 3) Mayo Systems Electrophysiology Laboratory, Department of Neurology, Mayo Clinic, Rochester, Minnesota, U.S.A.
% 4) Brno Epilepsy Center, Department of Neurology, St. Anne’s University Hospital and Medical Faculty of Masaryk University, Brno, Czech Republic
% 5) CEITEC – Central European Institute of Technology, Masaryk University, Brno, Czech Republic
% 6) Department of Physiology and Biomedical Engineering, Mayo Clinic, Rochester, Minnesota, U.S.A.
%===================================================================
%% Transfer Learning Code
% this code loads generalized CNN that is further retrained to the clinic
% specific model. 
clc
clear all
close all

% Load Generalized CNN from file
load('convnet.mat')

% Copy pretrained layers to the new neural network
layersTransfer=convnet.Layers(1:end-3);
%% Prepare dataset for training
rootFolder=fullfile('set_path_to_your_dataset_here');
%categories = {'60hz','noise', 'ok','patology'};
categories={'1','2','3','4'};
DATA = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames','FileExtensions','.mat');
DATA.ReadFcn = @customreader;
[TRAINING,VALIDATION,TESTING]= splitEachLabel(shuffle(DATA),1000,1000);

% print content of each dataset
countEachLabel(TRAINING)
countEachLabel(VALIDATION)
countEachLabel(TESTING)

%% Define structure of new convolutional neural network
layers = [
    layersTransfer
    fullyConnectedLayer(4,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];



%% Set training options
  options = trainingOptions('sgdm',...
                            'MaxEpochs',25,...
                            'LearnRateSchedule','piecewise',...
                            'LearnRateDropFactor',0.5,...
                            'LearnRateDropPeriod',7,...
                            'Momentum',0.9,...
                            'MiniBatchSize',200,...
                            'L2Regularization',0.001,...
                            'ExecutionEnvironment','multi-gpu',...
                            'Plots','training-progress',...
                            'ValidationData',VALIDATION,...
                            'ValidationFrequency',50);
  
  % NET TRAIN
  [convnet,info] = trainNetwork(TRAINING,layers,options);
  
%% Print confusion matrix for each dataset
YTest = classify(convnet,TRAINING);
TTest = TRAINING.Labels;
confmat=confusionmat(TTest,YTest)

YTest = classify(convnet,VALIDATION);
TTest = VALIDATION.Labels;
confmat=confusionmat(TTest,YTest)

YTest = classify(convnet,TESTING);
TTest = TESTING.Labels;
confmat=confusionmat(TTest,YTest)