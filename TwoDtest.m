% TestWaldBoostExample

% load data set
%load alldata features Y
% features=features(700:1300,:);
% Y=Y(700:1300);
% load Features-7000.mat features Y
%features=features(700:1300,:);
%Y=Y(700:1300);

features = [ mvnrnd([1 1],[1 0; 0 1],100); mvnrnd([-2 -2],[2 0; 0 2],100) ];
Y = [ones(1,100) ones(1,100)-2];

trainingRate=0.5;     % percentage of training part of overall data   
testTimes=1;          % test times
T=10;                % test iteration for every test


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


