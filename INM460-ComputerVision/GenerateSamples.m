% Generate test samples

searchpath = "../images/Processeds/";
savepath = "../images/model/sets/50-samples/";
samplelabelsize = 8;
samplefilesize = 50;
imagesizes = [227 128 70];
chromascheme = ["rgb" "grayscale"]
fnGetSampleImages(searchpath, savepath, samplelabelsize, samplefilesize, imagesizes, chromascheme);

savepath = "../images/model/sets/100-samples/";
samplefilesize = 100;
fnGetSampleImages(searchpath, savepath, samplelabelsize, samplefilesize, imagesizes, chromascheme);

savepath = "../images/model/sets/200-samples/";
samplefilesize = 200;
fnGetSampleImages(searchpath, savepath, samplelabelsize, samplefilesize, imagesizes, chromascheme);
