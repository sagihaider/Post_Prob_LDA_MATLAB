# Post_Prob_LDA_MATLAB
clear
load iris_dataset.mat 

X = irisInputs';
[~, y] = max(irisTargets); y = y'; % Transform target 

% Train Classifier
lda_clf =  fitcdiscr(X,y,'DiscrimType','linear'); % Classifier

% Get posterior probability
[label,post_probs] = predict(lda_clf,X); % Predicted probability for training data
disp(post_probs(52,:)) % Display predicted probability for datum 52
