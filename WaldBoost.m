% WaldBoost algorithm
% WaldBoostClassfy            Classify the training sample
% searchBestWeakLearner       Get the best weak learner
% trainWaldBoostLearner       Waldboost training process
% testWaldBoostLearner        Waldboost testing process
% testWaldBoost               Generate data sample and call Waldboost to train and test
% WaldBoost                   WaldBoost train and test entry
% 
% Input:
% trainX  training set
% trainY  training set tag
% T       total training iteration
% testX   test set
% testY   test set tag
% 
% 
% Output��
% WaldBoostInfo      cell structure, 1 * testTimes 
%                   Every element��
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
% costTime			cost time
% 
function [WaldBoostInfo]=WaldBoost(trainX,trainY,T,testX,testY)
[Hypothesis,AlphaT,trainErrorRate,trainOverallErrorRate,costTime,trainTPRate,trainFPRate]=trainWaldBoostLearner(trainX,trainY,T);
[testErrorRate,testOverallErrorRate,testTPRate,testFPRate]=testWaldBoostLearner(testX,testY,Hypothesis,AlphaT,T);

WaldBoostInfo.Hypothesis=Hypothesis;         % Weaker learner
WaldBoostInfo.AlphaT=AlphaT;                 % Weight for each weaker learner
WaldBoostInfo.trainError=trainErrorRate;     % training error over only classifed data
WaldBoostInfo.trainOverallError=trainOverallErrorRate;	% training error over all classified and unclassified data 
WaldBoostInfo.trainTPRate=trainTPRate;       % True positive rate over training set
WaldBoostInfo.trainFPRate=trainFPRate;       % False positive rate over training set
WaldBoostInfo.testError=testErrorRate;       % testing error over only classifed data
WaldBoostInfo.testOverallError = testOverallErrorRate;
WaldBoostInfo.testTPRate=testTPRate;         % True positive rate over test set
WaldBoostInfo.testFPRate=testFPRate;         % False positive rate over test set
WaldBoostInfo.costTime=costTime;             % cost time
