% Train networks, given parameters

%% CNN

datapath = [ "../images/model/sets/50-samples/" ...
            "../images/model/sets/100-samples/" ...
            "../images/model/sets/200-samples/"];
numFiles = [50 100 200];
split = 0.9; % 90% for training, 10% for testing

imagesizes = [227 128 70];
chromascheme = ["rgb" "grayscale"];

for r=1:length(datapath)
    for k=1:length(chromascheme)
        if chromascheme(k) == "rgb"
            numChannels = 3;
        else
            numChannels = 1;
        end
        for j=1:length(imagesizes)
            strdir = strcat(datapath(r), chromascheme(k), '/', num2str(imagesizes(j)), '/');
            [net, accuracy] = fnTrainCNN(strdir, numFiles(r), numChannels, split);
            msg = strcat(num2str(numFiles(r)), ",", num2str(imagesizes(j)), "x", num2str(imagesizes(j)) , ...
                "," , chromascheme(k), "," , num2str(accuracy));
            disp(msg);
        end
    end
end

%% HOG SVM
datapath = [ "../images/model/sets/50-samples/grayscale/" ...
            "../images/model/sets/100-samples/grayscale/" ...
            "../images/model/sets/200-samples/grayscale/"];
        
processSVM = 0;
split = 0.9;
% not happening hogpoints 36 
hogparams(1).hogpoints = 1500; hogparams(1).cellsize = [50 50];
hogparams(2).hogpoints = 2000; hogparams(2).cellsize = [25 25];
hogparams(3).hogpoints = 2500; hogparams(3).cellsize = [10 10];
% size in ../images/surf_grayscale/
imagesizes = [227 128 70];

for r=1:length(datapath)
    for k=1:length(hogparams)
        for j=1:length(imagesizes)
            strdir = strcat(datapath(r), '/', num2str(imagesizes(j)), '/');
            [HOGSVMMdl, HOGSVMAccuracy, SVMAccuracy] = ...
            fnCreateHOGSVMClassifier(strdir, hogparams(k).hogpoints, hogparams(k).cellsize, processSVM, split);
            scz = ['[' num2str(hogparams(k).cellsize) ']'];
            msg = strcat(num2str(numFiles(r)), ",", num2str(imagesizes(j)), "x", num2str(imagesizes(j)) , ...
                "," , num2str(hogparams(k).hogpoints) , "," , string(scz) , "," , num2str(HOGSVMAccuracy));
            disp(msg);
        end
    end
end

%% SURF SVM

imagesizes = [227 128 70];
SURFPointsSave = [40 50 60];

for r=1:length(datapath)
    for j=1:length(imagesizes)
        for k=1:length(SURFPointsSave)

            strdir = strcat(datapath(r), '/', num2str(imagesizes(j)), '/');
            [SURFSVMMdl, SURFSVMAccuracy, SVMAccuracy] = ...
            fnCreateSURFSVMClassifier(strdir, SURFPointsSave(k), processSVM, split);

            msg = strcat(num2str(numFiles(r)), ",", num2str(imagesizes(j)), "x", num2str(imagesizes(j)) , ...
                "," , num2str(SURFPointsSave(k)), ","  , num2str(SURFSVMAccuracy));
            disp(msg);
        end
    end
end
