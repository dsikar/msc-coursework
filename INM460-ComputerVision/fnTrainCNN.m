%fnTrainCNN(datapath, numFiles, numChannels, split)
%   Train a convolutional neural network
%   Inputs
%   datapath - path to folders containing labelled images
%   numFiles - number of files available
%   numChannels - number of colour channels
%   split - percentage of files used for training
%   Output:
%        net - trained convolutional neural network
%   Example:
%   >> datapath = '../images/Processed/';
%   >> numFiles = 50;
%   >> numChannels = 3; % rgb
%   >> split = 0.9; % 90% for training, 10% for testing
%   >> [net, accuracy] = fnTrainCNN(datapath, numFiles, numChannels, split)
%   >> img = imread('../images/Processed/22/75_14.JPG');
%   >> [class, err] = classify(net,img);
%   >> disp(class);
function [net, accuracy] = fnTrainCNN(datapath, numFiles, numChannels, split)
% ================ Start code ================
    % From Lab 8 - CNNs and OCR

    % Specify Training and Validation Sets
    p = numFiles * split;
    p = round(p);

    digitDatasetPath = datapath;
    imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

    % Calculate the number of images in each category
    labelCount = countEachLabel(imds);
    % Get a total to randomly choose from
    totalCount =  sum(table2array(labelCount(:,2)));
    % Get a class count
    classCount = size(labelCount.Label);
    % Convert row count to integer
    classCount = int8(classCount(1));

    % Check the size of the first image in digitData
    img = readimage(imds,1);
    [rows cols] = size(img);

    % help splitEachLabel
    [imdsTrain,imdsValidation] = splitEachLabel(imds,p,'randomized');

    % Define the convolutional neural network architecture.
    layers = [
        imageInputLayer([rows rows numChannels])

        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(classCount)
        softmaxLayer
        classificationLayer];

    % Specify Training Options % increasing number of epochs
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',imdsValidation, ...
        'ValidationFrequency',30, ...
        'Verbose',false, ...
        'ExecutionEnvironment','cpu', ...
        'Plots','training-progress');

    % Train Network Using Training Data
    net = trainNetwork(imdsTrain,layers,options);

    % Classify Validation Images and Compute Accuracy
    YPred = classify(net,imdsValidation);
    YValidation = imdsValidation.Labels;
    accuracy = sum(YPred == YValidation)/numel(YValidation); 

    % save network
    % dan_net4 = net;
    % save dan_net4
% ================ Start code ================
end