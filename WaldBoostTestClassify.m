function [predictOutput]=WaldBoostTestClassify(Samples,Y,Hypothesis,AlphaT,T,varargin)
error(nargchk(6,7,nargin));        % check input
iptcheckinput(Samples,{'numeric'},{'2d','real','nonsparse'}, mfilename,'Samples',1);
iptcheckinput(Hypothesis,{'numeric'},{'2d','real','nonsparse'}, mfilename,'Hypothesis',2);
iptcheckinput(AlphaT,{'numeric'},{'row','nonempty','real'},mfilename, 'AlphaT',3);

cntSamples=size(Samples,1);        % sample size
boostthresh=0.0;                   % Adaboost threshold(not used here)
if( nargin>5 )                     % set the threshold
    boostthresh=varargin{1};
end
iptcheckinput(T,{'numeric'},{'row','nonempty','integer'},mfilename, 'T',4);
if( length(T) > 1 )              % T should be a number
    error(['T should be a number']);
end
iptcheckinput(boostthresh,{'numeric'},{'row','nonempty','real'},mfilename, 'boostthresh',5);
if( length(boostthresh) > 1 )     % boostthresh should be a number
    error(['boostthresh should be a number']);
end

predictOutput=zeros(1,cntSamples); % predict output for each sample
predictConfidence=zeros(1,cntSamples); % confidence(not used here)
weakLearnerOutput = zeros(T,cntSamples);
interOutput = zeros(1,cntSamples);

Hypothesis=Hypothesis(1:T,:);      % Total T weak learner

AlphaT=AlphaT(1:T);                % Weight for weak learners

if(boostthresh == 1) % when the last round of classificaiton
  for i=1:T
    predout = WaldBoostWeakLearnerClassfy(Samples,Y,Hypothesis(i,:));
    weakLearnerOutput(i,:) = predout;
  end
  AlphaTMatrix = repmat(AlphaT',[1 cntSamples]);
  weakLearnerOutput = AlphaTMatrix.*weakLearnerOutput;
  interOutput = sum(weakLearnerOutput,1);
  predictOutput(interOutput<0) = -1;
  predictOutput(interOutput>=0) = 1;

  return
end

if(Hypothesis(T,4) == NaN && Hypothesis(T,5) == NaN)
  return
end
% calculate the strong classifier f(x) response.
for i=1:T
  predout = WaldBoostWeakLearnerClassfy(Samples,Y,Hypothesis(i,:));
  weakLearnerOutput(i,:) = predout;
end
AlphaTMatrix = repmat(AlphaT',[1 cntSamples]);
weakLearnerOutput = AlphaTMatrix.*weakLearnerOutput;
interOutput = sum(weakLearnerOutput,1);
% do waldboost classify.
  if(Hypothesis(T,1) ~= NaN)
    predictOutput(interOutput < Hypothesis(T,1)) = -1;
  end
  if(Hypothesis(T,2) ~= NaN)
    predictOutput(interOutput > Hypothesis(T,2)) = 1;
  end
