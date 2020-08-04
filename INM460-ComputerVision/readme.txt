INM460 Computer Vision Coursework
Daniel Sikar MSc Data Science PT2

README

**************************************
** Valid ways to call RecogniseFace **
**************************************

Preparing the environment

1. Unpack the contents of the zip file. 
2. Download the HOG-SVM model (HOGSVMMdl_70x70.mat) from google drive using link https://bit.ly/3f6oHqe and place in same directory as unpacked zip file contents.
2. Open Matlab and navigate to the unpacked files directory.
3. From the command prompt make sure you are in the working directory by listing the RecogniseFace function file
>> ls RecogniseFace.m

RecogniseFace.m  


The different options are available by running:
>> P = RecogniseFace('class5.jpg', 'SURF', 'SVM', 0)
>> P = RecogniseFace('class5.jpg', 'HOG', 'SVM', 0)
>> P = RecogniseFace('class5.png', '', 'CNN', 0)
And with creative mode:
>> P = RecogniseFace('class5.jpg', 'SURF', 'SVM', 1)
>> P = RecogniseFace('class5.jpg', 'HOG', 'SVM', 1)
>> P = RecogniseFace('class5.png', '', 'CNN', 1)

For convenience, file class5.jpg is provided. Also, script RunRecogniseFace.m is provided with all valid options as listed.

DELIVERABLES

In the zip file are included, in addition to this readme file, source code comprising all MATLAB .m files, the INM460_Computer_Vision_coursework_report.pdf report, ScreenCaptureVideo.mp4 and two trained models (dan_net_70_rgb.mat,  and SURFSVMMdl_70px_50pt.mat). Please don't forget to download HOG-SVM model.
