%fnCreateHOGSVMClassifier(datapath, HOGPointsSave, cellsize, processSVM, split)
% Generate HOG-SVM classifier, compute accuracy and plot confusion matrix
%   Inputs:
%       datapath - image folder directory path
%       HOGPointsSave - number of points to save
%       cellsize - h x w array size of a HOG cell in pixels
%
%   Note: HOGPointsSave is a function of cellsize. Some valid options:
%   HOGPointsSave = 36; % cellsize [50 50]
%   HOGPointsSave = 324; % cellsize [25 25]
%   HOGPointsSave = 2916; % cellsize [10 10]
%   To determine additional HOGPointsSave values, choose a cellsize, run
%   the function inserting a breakpoint on line:
%           xTrainHOG(:,i) = hog1';
%   Display the size of hog1
%           disp(hog1)
%   Rerun the function with that quantity for HOGPointsSave.
%
%   processSVM - boolean flag, if true, also compute SVM accuracy
%   split - training/testing split e.g. 0.9 = 90% train 10% test
%   Outputs: 
%       HOGSVMMdl - SVM classifier
%       HOGSVMAccuracy - SURF SVM classifier accuracy
%       SVMAccuracy - vanilla SVM classifier accuracy
%   Example:
%   >> datapath = '../images/surf_grayscale/';
%   >> HOGPointsSave = 2916;
%   >> cellsize = [10 10];
%   >> processSVM = 0;
%   >> split = 0.9;
%   >> [HOGSVMMdl, HOGSVMAccuracy, SVMAccuracy] = ...
%   >> fnCreateHOGSVMClassifier(datapath, HOGPointsSave, cellsize, processSVM, split)
function [HOGSVMMdl, HOGSVMAccuracy, SVMAccuracy] = fnCreateHOGSVMClassifier(datapath, HOGPointsSave, cellsize, processSVM, split)

    % Some sections of code adapted from INM460 lab 06.

    SVMAccuracy = 0;

    % digitDatasetPath = '../images/surf_grayscale/';
    imds = imageDatastore(datapath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

    % convert labels from categorical to numeric data, so model may be
    % saved later
    d = string(imds.Labels);
    imds.Labels = double(d);
    
    [xTrainImages, xTestImages] = splitEachLabel(imds,split,'randomized');

    % Load the training data into memory
    % [xTrainImages, tTrain] = digittrain_dataset;
    % Display some of the training images
    %clf
    %for i = 6:25
    %    subplot(4,5,i-5);
        % imshow(xTrainImages{i});
    %    imshow(xTrainImages.readimage(i));
    %end

    % Get the number of pixels in each image
    [imageWidth, imageHeight] = size(xTrainImages.readimage(1));
    inputSize = imageWidth*imageHeight; % Load the test images
    % read all images
    imgtrain = readall(xTrainImages);
    % Turn the training images into vectors and put them in a matrix
    xTrain = zeros(inputSize,numel(imgtrain));
    % Points to keep - turned into function arguments
    % HOGPointsSave = 36; % cellsize [50 50]
    % HOGPointsSave = 324; % cellsize [25 25]
    %HOGPointsSave = 2916; % cellsize [10 10]
    %cellsize = [10 10];
    % Vector storage point size x 64 to hold partial featuresOriginal matrix
    xTrainHOG = zeros(HOGPointsSave,numel(imgtrain));
    for k = 1:numel(imgtrain)
        if processSVM == 1
            xTrain(:,k) = imgtrain{k}(:); % (:) effectively stacks all columns
        end
        % extract HOG features
        [hog1,visualization] = extractHOGFeatures(imgtrain{k},'CellSize',cellsize);
        % get the size
        [rows cols] = size(hog1);
        % save
        f = hog1(:,1:cols);
        if HOGPointsSave <= cols
            % drop columns/reshape feature vector
            f = f(:,1:HOGPointsSave);
            % flag - display to adjust size empirically
            msg = strcat('Index ', string(k), " dropped ", string(cols-HOGPointsSave), " columns.");
            %disp(msg)
        else
            % pad missing columns with zeros
            padsize = HOGPointsSave - cols;
            f = [f, zeros(1,padsize)];
        end
        xTrainHOG(1:HOGPointsSave,k) = f';
    end

    imgtest = readall(xTestImages);
    % Turn the test images into vectors and put them in a matrix
    xTest = zeros(inputSize,numel(imgtest));
    xTestHOG = zeros(HOGPointsSave,numel(imgtest));

    for i = 1:numel(imgtest)
        if processSVM == 1
            xTest(:,i) = imgtest{i}(:);
        end
        % extract HOG features
        [hog1,visualization] = extractHOGFeatures(imgtest{i},'CellSize',cellsize);
        % get the size
        [rows cols] = size(hog1);
        % save
        f = hog1(:,1:cols);
        if HOGPointsSave <= cols
            % drop columns/reshape feature vector
            f = f(:,1:HOGPointsSave);
            % flag - display to adjust size empirically
            msg = strcat('Index ', string(k), " dropped ", string(cols-HOGPointsSave), " columns.");
            %disp(msg)
        else
            % pad missing columns with zeros
            padsize = HOGPointsSave - cols;
            f = [f, zeros(1,padsize)];
        end
        xTestHOG(1:HOGPointsSave,i) = f';
    end

    SVMMdl = fitcecoc(xTrain',xTrainImages.Labels); %SVM MULTICLASS TRAINER
    % SURF SVM
    HOGSVMMdl = fitcecoc(xTrainHOG',xTrainImages.Labels);
    if processSVM == 1
        labelsOut = predict(SVMMdl,xTest');
    end
    % Predict HOG Labels
    HOGlabelsOut = predict(HOGSVMMdl,xTestHOG');
    if processSVM == 1
        ConfMatTest = confusionmat(xTestImages.Labels,labelsOut);
    end
    % SURF CM
    HOGConfMatTest = confusionmat(xTestImages.Labels,HOGlabelsOut);

    if processSVM == 1
        SVMAccuracy = 1 - (numel(imgtest) - sum(diag(ConfMatTest))) / numel(imgtest);
    end
    HOGSVMAccuracy = 1 - (numel(imgtest) - sum(diag(HOGConfMatTest))) / numel(imgtest);

    % HOG Confusion Matrix
    plotconfusion(categorical(xTestImages.Labels),categorical(HOGlabelsOut));

    % save model
    % saveCompactModel(HOGSVMMdl, 'HOGSVMMdl');
    
end