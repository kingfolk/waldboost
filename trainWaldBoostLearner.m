% WaldBoost algorithm
% WaldBoostClassfy            Classify the training sample
% searchBestWeakLearner       Get the best weak learner
% trainWaldBoostLearner       Waldboost training process
% testWaldBoostLearner        Waldboost testing process
% testWaldBoost               Generate data sample and call Waldboost to train and test
% WaldBoost                   WaldBoost train and test entry
% 
% Input:
% X            train set
%               integral patch 
%               cntSamples*768 matrix from cntSamples*16*16*3
% Y            correct train set tag
% T            training iteration time
% cntSelectFeatures 
%              可选参数，需要训练指定数量的特征的数量
%              若输入4个参数，则表示不会依照训练轮数为T的规则
%              此时将一直训练，直至分类器训练出指定的特征
%              在这种情况下 costTime 指提取单个特征所获得的时间
% 
% Output：
% Hypothesis     Weak learner
%                      thres   sign   feature
%                   -16.4151    1       27
%                     0.0073    1       291
%                     0.4482   -1       14
%                     0.0540    1       315
% 
% AlphaT         Weight for each weak learner
% trainErrorRate training error. 1*T vector
% costTime       cost time over T iteration
%
% call example：
% [Hypothesis,AlphaT,trainErrorRate,OverallErrorRate,costTime,TPRate,FPRate]=trainWaldBoostLearner(X,Y,T)
% [Hypothesis,AlphaT,trainErrorRate,costTime]=trainWaldBoostLearner(X,Y,T,cntSelectFeatures)
% 
%
function [Hypothesis,AlphaT,trainErrorRate,OverallErrorRate,costTime,TPRate,FPRate]=trainWaldBoostLearner(X,Y,T,varargin)
error(nargchk(3,4,nargin)); % 必须输入3-4个参数,否则中止程序
iptcheckinput(X,{'numeric'},{'2d','real','nonsparse'}, mfilename,'X',1);
iptcheckinput(Y,{'logical','numeric'},{'row','nonempty','integer'},mfilename, 'Y', 2);
iptcheckinput(T,{'numeric'},{'row','nonempty','integer'},mfilename, 'T',3);
if( length(T) > 1 )              % 指定训练轮数的参数T长度应为1（不能为向量）
    error(['T should be a number']);
end

[cntSamples,cntFeatures]=size(X); % cntSamples  train set size
inverseControl=0;           % loop control
cntSelectFeatures=0;        % feature number to iterate

if( nargin>3 )              % 4 input
    cntSelectFeatures=varargin{1};
    inverseControl=1;
    iptcheckinput(cntSelectFeatures,{'numeric'},{'row','nonempty','integer'},mfilename, 'cntSelectFeatures',4);
    if( length(cntSelectFeatures) > 1 ) % feature number should not be vector
        error(['(cntSelectFeatures) should be a number']);
    end
    if( cntSelectFeatures>=cntFeatures )
        error('(cntSelectFeatures) too big！');
    end
end
if( cntSamples~=length(Y) ) % check if match
    error('error cntSamples length does not match tag length') ;
end

computeCosttimeFlag=1;      % counting flag. count time if equal to 1
if(computeCosttimeFlag==1)
    tic
end

X=ceil(X*10000)/10000;          % rounding X
positiveCols=find(Y==1);        % cols for tag = 1
negativeCols=find(Y==-1);        % cols for tag = -1
if(length(positiveCols)==0)     % check not equal to 0
    error('positive sample size 0.');
end
if(length(negativeCols)==0)     % check not equal to 0
    error('negative sample size 0.');
end

weight=ones(1,cntSamples);       % initialize weight for samples
weight(positiveCols)=1/(2*length(positiveCols));
weight(negativeCols)=1/(2*length(negativeCols));
Hypothesis=zeros(T,20);           % initialize output value
AlphaT=zeros(1,T);              
trainErrorRate=zeros(1,T);      
OverallErrorRate=zeros(1,T);
costTime=zeros(1,T);  
TPRate = zeros(1,T);             
FPRate = zeros(1,T);                      

trainOutput=zeros(1,cntSamples); % intermediate vector: store the output from temp learner  
h=zeros(1,cntSamples);           % intermediate vector: store the output of best weaker learner      
Decision=zeros(1,cntSamples);

t=1; 
curFeaSize=0;                   % size of features which have been used for training
featureCascadeNum = 50;
while(true)                     
    minError=cntSamples;        % Misclassified initialization：cntSamples
    %weight=weight/(sum(weight));% Normalize weight
    undecided_idx = find(Decision==0); % 1. remove decided samples
    %undecided_idx = [1:cntSamples];    % 2. keep all samples
    [t,size(undecided_idx,2)]           %debug info
    samples_X = X(undecided_idx,:);
    samples_Y = Y(undecided_idx);
    samples_weight = weight(undecided_idx);
    samples_weight=samples_weight/(sum(samples_weight));
    %[HypothesisOut predout] = BestAdaBoostWeakLearnerManual(samples_X(:,:),samples_Y,samples_weight);
    [HypothesisOut predout] = BestAdaBoostWeakLearner(samples_X(:,:),samples_Y,samples_weight,featureCascadeNum);
    lengthofhyp = length(HypothesisOut);
    Hypothesis(t,1:lengthofhyp) = HypothesisOut;
    %h=WaldBoostWeakLearnerClassfy(samples_X,Hypothesis(t,:));   % classify with best learner

    errorRate=sum(samples_weight(find(predout~=samples_Y))); 
    errorRate             % calculate the misclassifed ratio
    AlphaT(t)=log10((1-errorRate)/(errorRate+eps));    % calculate the weight         
    if(errorRate>eps)                                  % weight down the correctly classified samples
        samples_weight(find(predout==samples_Y))=samples_weight(find(predout==samples_Y))*(errorRate/(1-errorRate));                                     
    end
    weight(undecided_idx) = samples_weight(:);
    % at t iteration, get the error rate and threshold for current strong learner
    %[trainOutput,thresA,thresB]=WaldBoostClassfy(samples_X,samples_Y,Hypothesis,AlphaT,t);
    [trainOutput,thresA,thresB]=WaldBoostCl(samples_X,samples_Y,Hypothesis,AlphaT,t,t==T);
    idx = find(trainOutput~=0);
    newdecided_idx = undecided_idx(idx);
    Decision(newdecided_idx) = trainOutput(idx);

    [curErrorRate,curOverallErrorRate,curTPRate,curFPRate]=calPredictErrorRate(Y,Decision);
    Hypothesis(t,1:2) = [thresA thresB];
    %t
    %curErrorRate
    %size(curErrorRate,2)
    %if(size(curErrorRate,2)>1)
    %    curErrorRate = curErrorRate(1,1);
    %end
    %whos curErrorRate
    trainErrorRate(t) = curErrorRate;
    OverallErrorRate(t) = curOverallErrorRate;
    TPRate(t) = curTPRate;
    FPRate(t) = curFPRate;
   
    if(inverseControl==0)            % control if 4 input
        if(computeCosttimeFlag==1)
            costTime(t)=toc;
        end
        if(t>=T)                     % reach required iteration number                    
             break;
        end
    else                             % feature control
        [SelectFeaNo]=analysisHypothesisFeature(Hypothesis(1:t,:),0);
        if( length(SelectFeaNo)>curFeaSize )
            curFeaSize=length(SelectFeaNo);
            if( computeCosttimeFlag==1 )
                costTime(curFeaSize)=toc;
            end
        end
        if( curFeaSize>=cntSelectFeatures )% reach feature size
            break;
        end
    end
    t=t+1;
end

if(computeCosttimeFlag==1)
    costTime=costTime(find(costTime~=0));% ouput cost time
else
    costTime=0;
end

if(t<T)     
    Hypothesis=Hypothesis(1:t,:);      % strong classifier
    AlphaT=AlphaT(1:t);                % strong classifier weight
    trainErrorRate=trainErrorRate(1:t);% training error
end

