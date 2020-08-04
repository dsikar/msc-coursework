disp('Running CNN...');
P = RecogniseFace('class5.jpg', '', 'CNN', 0)
disp('Running HOG-SVM...');
P = RecogniseFace('class5.jpg', 'HOG', 'SVM', 0)
disp('Running SURF-SVM...');
P = RecogniseFace('class5.jpg', 'SURF', 'SVM', 0)
disp('Running CNN Creative Mode...');
P = RecogniseFace('class5.png', '', 'CNN', 1)
disp('Running HOG-SVM Creative Mode...');
P = RecogniseFace('class5.jpg', 'HOG', 'SVM', 1)
disp('Running SURF-SVM Creative Mode...');
P = RecogniseFace('class5.jpg', 'SURF', 'SVM', 1)

% Final stats code
% Class - unlabelled
P = RecogniseFace('class1.jpg', '', 'CNN', 0);
P = RecogniseFace('class2.jpg', '', 'CNN', 0);
P = RecogniseFace('class3.jpg', '', 'CNN', 0);
P = RecogniseFace('class4.jpg', '', 'CNN', 0);
P = RecogniseFace('class5.jpg', '', 'CNN', 0);
% 14/101 ~ 0.1386 Accuracy

% HOG
P = RecogniseFace('class1.jpg', 'HOG', 'SVM', 0);
P = RecogniseFace('class2.jpg', 'HOG', 'SVM', 0);
P = RecogniseFace('class3.jpg', 'HOG', 'SVM', 0);
P = RecogniseFace('class4.jpg', 'HOG', 'SVM', 0);
P = RecogniseFace('class5.jpg', 'HOG', 'SVM', 0);
% 13/101 ~ 0.1287 Accuracy

% SURF
P = RecogniseFace('class1.jpg', 'SURF', 'SVM', 0);
P = RecogniseFace('class2.jpg', 'SURF', 'SVM', 0);
P = RecogniseFace('class3.jpg', 'SURF', 'SVM', 0);
P = RecogniseFace('class4.jpg', 'SURF', 'SVM', 0);
P = RecogniseFace('class5.jpg', 'SURF', 'SVM', 0);
% 12/101 ~ 0.1188 Accuracy

% Individual - labelled
P = RecogniseFace('individual1.jpg', '', 'CNN', 0); 
P = RecogniseFace('individual2.jpg', '', 'CNN', 0);
P = RecogniseFace('individual3.jpg', '', 'CNN', 0);
P = RecogniseFace('individual4.jpg', '', 'CNN', 0);
P = RecogniseFace('individual5.jpg', '', 'CNN', 0);
% 5/5 ~ 1 Accuracy

% HOG
P = RecogniseFace('individual1.jpg', 'HOG', 'SVM', 0);
P = RecogniseFace('individual2.jpg', 'HOG', 'SVM', 0);
P = RecogniseFace('individual3.jpg', 'HOG', 'SVM', 0);
P = RecogniseFace('individual4.jpg', 'HOG', 'SVM', 0);
P = RecogniseFace('individual5.jpg', 'HOG', 'SVM', 0);
% 5/5 ~ 1 Accuracy

% SURF
P = RecogniseFace('individual1.jpg', 'SURF', 'SVM', 0);
P = RecogniseFace('individual2.jpg', 'SURF', 'SVM', 0);
P = RecogniseFace('individual3.jpg', 'SURF', 'SVM', 0);
P = RecogniseFace('individual4.jpg', 'SURF', 'SVM', 0);
P = RecogniseFace('individual5.jpg', 'SURF', 'SVM', 0);
% 5/5 ~ 1 Accuracy