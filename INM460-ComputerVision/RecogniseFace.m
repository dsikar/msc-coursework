%RecogniseFace(I, featureType, classifierType, creativeMode)
%RecogniseFace - Recognise face based on feature and classifier type (H1 line)
% Recognise faces in image.
% Inputs:
%    I - image path 
%    featureType - SURF, HOG
%    classifierType - SVM, MLP, CNN
%       Implemented featureType/ClassifierType:
%       'SURF', 'HOG'
%       'SURF', 'SVM'
%       '', 'CNN'
%    creativeMode - boolean flag, if set face image is modified creatively
% Outputs:
%    P - matrix P describing the student(s) present in an RGB image I. 
%    P is a matrix of size N x 3, where N is the number of people detected 
%    in the image. Example, if the function detects two students in the 
%    image, with student with ID=03 having his/her face centred at position 
%    [144, 153], and student with ID=13 at position [312,123], the P matrix 
%    would be
%    P =
%       03 144 153
%       13 312 123
%
% Example: 
%    SURF, HOG, no creative mode
%    P = RecogniseFace('class.jpg', 'SURF', 'SVM', 0)
%    P = RecogniseFace('class.jpg', 'HOG', 'SVM', 0)
%    CNN, creative mode
%    P = RecogniseFace('class.png', '', 'CNN', 1)
%
% Author: Daniel Sikar - MSc Data Science - PT2
% SMCSE - City, University of London - daniel.sikar@city.ac.uk
function P = RecogniseFace(I, featureType, classifierType, creativeMode)  
%------------- BEGIN CODE --------------
    
    % Validation
    % 1. Image
    try
        I = fnSetImageUpright(I); 
    catch
        error('Invalid filepath or file. See ''help RecogniseFace''.');
        return;
    end   
    % 2. Classifier and feature types validation
    % SURF-SVM, HOG-SVM or CNN
    if ~( (strcmp(featureType,'SURF') & strcmp(classifierType,'SVM') ) ...
      |   ( strcmp(featureType,'HOG') & strcmp(classifierType,'SVM') ) ...
      |   ( strcmp(featureType,'') & strcmp(classifierType, 'CNN') ) )
  
        error('Invalid feature-classifier. See ''help RecogniseFace''.');
        return;
        
    end;

    % load classifiers
    if classifierType == "CNN"
        cnnet = load('dan_net_70_rgb');
        gLabel = "CNN";
    elseif featureType == "HOG"
        HOGPointsSave = 2500;
        cellsize = [10 10]; % set when training model
        HOGSVMMdl = load('HOGSVMMdl_70x70.mat'); 
        gLabel = "HOG-SVM";
    elseif featureType == "SURF"
        SURFPointsSave = 50; % set when training model
        SURFSVMMdl = load('SURFSVMMdl_70px_50pt.mat');  
        gLabel = "SURF-SVM";        
    end
    % adjust photo classifier label
    if creativeMode == 1
        gLabel = strcat(gLabel, " Creative Mode");
    end

    % minimum expected face size in photo
    minsize = [85 85];    
    FaceDetector = vision.CascadeObjectDetector('MinSize', minsize);
    % get bounding boxes
    bbox = step(FaceDetector, I);
    [rows,cols] = size(bbox(:,:));
    % initialise return array
    P = [];
    
    for i = 1:rows
        % 1. draw rectangles
        a = bbox(i, 1);
        b = bbox(i, 2);
        c = bbox(i, 3);
        d = bbox(i, 4);

        % Crop
        c2 = a+c;
        d2 = b+d;
        imgcrop = I(b:d2, a:c2, :);
        % scale
        % skip for now
        % Convert to grayscale and resize, as per training/testing dataset
        % imgcrop = rgb2gray(imgcrop);
        % The size used to train our classifiers and neural network
        side = 70;
        imgcropclass = imresize(imgcrop,[side side]);
        % imgcropclass = rgb2gray(imgcropclass);
        if classifierType == "CNN"
            % predict
            [myclass, err] = classify(cnnet.net,imgcropclass);
            % get confidence
            conf = max(err);
        elseif featureType == "HOG"
            imgcropclass = rgb2gray(imgcropclass);
            myclass = fnHOGSMVPredict(HOGSVMMdl, HOGPointsSave, cellsize, imgcropclass);            
        elseif featureType == "SURF"
            imgcropclass = rgb2gray(imgcropclass);
            myclass = fnSURFSMVPredict(SURFSVMMdl, SURFPointsSave, imgcropclass);;
        end 
        % Creative mode
        if creativeMode == 1
            try
                img = fnPandemicMode(imgcrop);
                I(b:d2, a:c2, :) = img;
            catch ME
                warning('Could not supply facemask for this image');
            end 
        end
        I = insertShape(I,'rectangle', [a b c d],'LineWidth',10, ...
           'Color', [3 252 219], 'Opacity',0.7);
        % set ID as not found
        ID = -1;
        conf_thresh = 0.50;
        % debug flag
        debug = 0;       
        if classifierType == "CNN"
            if conf >= conf_thresh 
                ID = str2double(string(myclass));
                if debug == 1
                    IDlabel = strcat(string(i), ': ', string(ID));
                else
                    IDlabel = string(ID);
                end
                %v = str2num(ID);
            end
        else
            IDlabel = string(myclass);
            ID = str2double(IDlabel);
            conf = -1; % N/A
        end       
        % ID print offset
        offset = 0; % pixels
        pos = [(a+offset) (b+offset*2)];
        
        I = insertText(I, pos,cellstr(IDlabel), 'FontSize',60, 'BoxOpacity',0, ...
            'Font', 'Consolas Bold', 'TextColor',[0 255 0]);
        % debug info
        msg = strcat('ID: ', string(ID), ', image: ', string(i), ', conf: ', ... 
            string(conf), ', size: ', string(c), 'x', string(d));
        %disp(msg);
        
        a = round(a + c/2);
        b = round(b + d/2);
        % ID = str2double(ID);
        P(i,:) = [ID, a, b];        
    end    
    pos = [20 20];
    % Label photo with classifier
    I = insertText(I, pos,cellstr(gLabel), 'FontSize',80, 'BoxOpacity',0, ...
    'Font', 'Consolas Bold', 'TextColor',[244 252 3]);
    %'Font', 'Consolas Bold', 'TextColor','green');
    figure; imshow(I);   
%------------- END OF CODE --------------
end