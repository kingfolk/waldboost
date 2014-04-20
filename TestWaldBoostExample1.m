% TestWaldBoostExample

% load data set
% load training_3400p_1984n feature_positive feature_negative
load training_3400p_integralpatch feature_positive feature_negative
pos_size = size(feature_positive,1);
neg_size = size(feature_negative,1);
Y = [ones(1,pos_size) ones(1,neg_size)-2];
halfsize = 3400;
features=[feature_positive(pos_size-halfsize+1:pos_size,:); feature_negative(neg_size-halfsize+1:neg_size,:)];
Y=Y(pos_size-halfsize+1:pos_size+halfsize);

trainingRate=0.5;     % percentage of training part of overall data   
testTimes=1;          % test times
T=100;                % test iteration for every test


[BoostInfomation]=testWaldBoost(features,Y,trainingRate,testTimes,T)


%return;
for i=1:length(BoostInfomation)
    BoostInfo=BoostInfomation{i};
    Hypothesis{i}=BoostInfo.Hypothesis;
    trainError{i}=BoostInfo.trainError;
    trainOverallError{i}=BoostInfo.trainOverallError;
    testError{i}=BoostInfo.testError;
    testOverallError{i}=BoostInfo.testOverallError;
    TP{i}=BoostInfo.testTPRate;
    FP{i}=BoostInfo.testFPRate;
    costTime{i}=BoostInfo.costTime;
end

figure(1002);hold on,
grid on,
xlabel('trainning iter');
ylabel(strcat('Boost classifier error rate ( testing',num2str(testTimes),'times ) '));
testingNum=ceil((1-trainingRate)*size(features,1)); % testing sample size
trainingNum=size(features,1)-testingNum;            % training sample size
title(strcat('Boost classifier error rate',' ( trainning',num2str(trainingNum),'times,testing',num2str(testingNum),'times )'));

testRange=1:T;
for i=1:testTimes
    plot(testRange,trainError{i}(testRange),'m-');
    plot(testRange,testError{i}(testRange),'c-');
end

legend('WaldBoost trainning error','WaldBoost testing error');

figure(1003);hold on,
grid on,
xlabel('trainning iter');
ylabel(strcat('Boost classifier error rate ( testing',num2str(testTimes),'times ) '));
testingNum=ceil((1-trainingRate)*size(features,1)); 
trainingNum=size(features,1)-testingNum;            
title(strcat('Boost Overall classifier error rate',' ( trainning',num2str(trainingNum),'times,testing',num2str(testingNum),'times )'));

testRange=1:T;
for i=1:testTimes
    plot(testRange,trainOverallError{i}(testRange),'m-');
    plot(testRange,testOverallError{i}(testRange),'c-');
end

legend('WaldBoost Overall trainning error','WaldBoost Overall testing error');

return;


