%fnHOGSMVPredict(HOGSVMMdl, HOGPointsSave, cellsize, img)
% Generate a HOG SVM prediction based hog points and cell size used to generate
% model. Assumes grayscale images are used.
% Inputs:
%   HOGSVMMdl - the classifier model
%   HOGPointsSave - number of points to save
%   cellsize - array size of a HOG cell in pixels
%
%   Note, HOGPointsSave and cellsize are used to generate HOG SVM classifier models and the
%   same values must be used for model predictions, i.e. if 2500 points and [10 10] cellsize
%   were used to generate model, the same values should be used for
%   predictions. See "help fnCreateHOGSVMClassifier" for more
%   information.
%
%   img - grayscale face image to predict label of
% Output:
%   label - the predicted face image label
% Example:
% >> img = imread('../images/model/sets/200-samples-final/grayscale/70/02/6_81.JPG');
% >> HOGPointsSave = 2500;
% >> cellsize = [10 10]
% >> HOGSVMMdl_70x70 = load('HOGSVMMdl_70x70.mat');
% >> myclass = fnHOGSMVPredict(HOGSVMMdl_70x70, HOGPointsSave, cellsize, img);
function label = fnHOGSMVPredict(HOGSVMMdl, HOGPointsSave, cellsize, I)

    % For compatibility with RecogniseFace, image is used, not image path, 
    % as cropped face images will be passed to function
    %I = imread(imgpath);
    % extract HOG features
    [hog1,visualization] = extractHOGFeatures(I,'CellSize',cellsize);
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
    label = predict(HOGSVMMdl.HOGSVMMdl_70x70, f);
end