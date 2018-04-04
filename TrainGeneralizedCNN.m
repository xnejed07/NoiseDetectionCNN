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
%% This script is used for training of generalized model
clc
clear all
close all

%% This dataset is used for training and testing
TRAINING_DATASET = fullfile('C:\Users\nejedly\Prace\Projekty\NoiseMix\DatasetFnusa');
categories = {'2', '3','4'};
TRAINING_DATASET = imageDatastore(fullfile(TRAINING_DATASET, categories), 'LabelSource', 'foldernames','FileExtensions','.mat');
TRAINING_DATASET= shuffle(TRAINING_DATASET);
TRAINING_DATASET.ReadFcn = @customreader;
[TRAINING_DATASET_test,TRAINING_DATASET_train]=splitEachLabel(TRAINING_DATASET,0.3);
countEachLabel(TRAINING_DATASET_test)
countEachLabel(TRAINING_DATASET_train)
%% This dataset is used only for validation, Gradients computation and backpropagation is not used on this dataset.
%Once validation error estimated by this dataset starts to increse, training process is stopped.

VALIDATION_DATASET=fullfile('C:\Users\nejedly\Prace\Projekty\NoiseMix\DatasetMayo');
categories = {'2', '3','4'};
VALIDATION_DATASET = imageDatastore(fullfile(VALIDATION_DATASET, categories), 'LabelSource', 'foldernames','FileExtensions','.mat');
VALIDATION_DATASET=shuffle(VALIDATION_DATASET);
VALIDATION_DATASET.ReadFcn = @customreader;
[VALIDATION_DATASET_valid,VALIDATION_DATASET_test]= splitEachLabel(VALIDATION_DATASET,1000,1000);
countEachLabel(VALIDATION_DATASET_valid)
countEachLabel(VALIDATION_DATASET_test)

%% TRAIN CNN
reset(gpuDevice(1))
reset(gpuDevice(2))

layers = [imageInputLayer([5 15000 1],'normalization','none')
     convolution2dLayer([5 50],50,'Stride', [1 10])
     batchNormalizationLayer
	 reluLayer()
     maxPooling2dLayer([1 2], 'Stride',[1 2])
     convolution2dLayer([1 150],50,'Stride',[1 1])
     batchNormalizationLayer
     reluLayer()
     maxPooling2dLayer([1 2], 'Stride',[1 2])
     dropoutLayer(0.5)
     fullyConnectedLayer(150)
     batchNormalizationLayer
     reluLayer()
     dropoutLayer(0.5)
     fullyConnectedLayer(25)
     batchNormalizationLayer
     reluLayer()
     dropoutLayer(0.5)
	 fullyConnectedLayer(3)
	 softmaxLayer()
	 classificationLayer()];
 
  options = trainingOptions('sgdm',...
                            'MaxEpochs',25,...
                            'LearnRateSchedule','piecewise',...
                            'LearnRateDropFactor',0.5,...
                            'LearnRateDropPeriod',7,...
                            'Momentum',0.9,...
                            'MiniBatchSize',200,...
                            'L2Regularization',0.001,...
                            'ExecutionEnvironment','gpu',...
                            'Plots','training-progress',...
                            'ValidationData',VALIDATION_DATASET_valid,...
                            'ValidationFrequency',500);
  % NET TRAIN
  [convnet,info] = trainNetwork(TRAINING_DATASET_train,layers,options);
    

%% Evaluate results

YTest = classify(convnet,TRAINING_DATASET_test);
TTest = TRAINING_DATASET_test.Labels;
confmat=confusionmat(TTest,YTest)

YTest = classify(convnet,VALIDATION_DATASET_test);
TTest = VALIDATION_DATASET_test.Labels;
confmat=confusionmat(TTest,YTest)






