% WaldBoost algorithm
% WaldBoostClassfy            Classify the training sample
% searchBestWeakLearner       Get the best weak learner
% trainWaldBoostLearner       Waldboost training process
% testWaldBoostLearner        Waldboost testing process
% testWaldBoost               Generate data sample and call Waldboost to train and test
% WaldBoost                   WaldBoost train and test entry
% 
% 
% Input:
% dataset      cntSamples * cntFeatures matrix
% Y            correct tag for each single sample
% trainingRate percentage of training set to dataset
% testTimes    testing times
% T            iteration times for each test
%
% Output£º
% WaldBoostInfo      cell structure, 1 * testTimes 
%                   Every element£º
% trainError        training error over only classifed data
% trainOverallError training error over all classified and unclassified data    
% testError         testing error over only classifed data
% testOverallError  testing error over all classified and unclassified data  
% trainTPRate       True positive rate over training set
% trainFPRate       False positive rate over training set
% testTPRate        True positive rate over test set
% testFPRate        False positive rate over test set
% Hypothesis        Weaker learner
% AlphaT            Weight for each weaker learner
% costTime          cost time
% 
function [WaldBoostInfo]=testWaldBoost(dataset,Y,trainingRate,testTimes,T)
WaldBoostInfo=cell(1,testTimes);
for curTestTime=1:testTimes
    cntSamples=size(dataset,1); % size of data set
    [trainingIndexs,testingIndexs]=generateTrainTestSamples(cntSamples,trainingRate);
                                              % random distribute train and test set over data set
    trainX=dataset(trainingIndexs,:);         % get training data
    trainY=Y(trainingIndexs);                 % get training data tag
    testX=dataset(testingIndexs,:);           % get test data
    testY=Y(testingIndexs);                   % get test data tage

    disp(strcat('testing WaldBoost--Running No.',num2str(curTestTime),', Over ',num2str(testTimes),':'));  

    [curWaldBoostInfo]=WaldBoost(trainX,trainY,T,testX,testY);
    WaldBoostInfo{curTestTime}=curWaldBoostInfo;

end
disp('----------------------------------------------');


