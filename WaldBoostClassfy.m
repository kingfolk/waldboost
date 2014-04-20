%function [predictOutput,ErrorRate,OverallErrorRate,TPrate,FPrate,thresA,thresB]=WaldBoostClassfy(Samples,Y,Hypothesis,AlphaT,T,varargin)
function [predictOutput,thresA,thresB]=WaldBoostClassfy(Samples,Y,Hypothesis,AlphaT,T,varargin)
error(nargchk(5,6,nargin));        % check input
iptcheckinput(Samples,{'numeric'},{'2d','real','nonsparse'}, mfilename,'Samples',1);
iptcheckinput(Hypothesis,{'numeric'},{'2d','real','nonsparse'}, mfilename,'Hypothesis',2);
iptcheckinput(AlphaT,{'numeric'},{'row','nonempty','real'},mfilename, 'AlphaT',3);

cntSamples=size(Samples,1);        % number of samples
boostthresh=0.0;                   % AdaBoost thres for weak learner
if( nargin>5 )                     % to set the thres
    boostthresh=varargin{1};
end
iptcheckinput(T,{'numeric'},{'row','nonempty','integer'},mfilename, 'T',4);
if( length(T) > 1 )              % not a vector
    error(['T should be a number.']);
end
iptcheckinput(boostthresh,{'numeric'},{'row','nonempty','real'},mfilename, 'boostthresh',5);
if( length(boostthresh) > 1 )     % not a vectore
    error(['boostthresh should be a number.']);
end

predictOutput=zeros(1,cntSamples); % predict output by current strong classifier
predictConfidence=zeros(1,cntSamples); % confidence(not used here)

Hypothesis=Hypothesis(1:T,:);      
AlphaT=AlphaT(1:T);               
weakLearnerOutput = zeros(T,cntSamples);
interOutput = zeros(1,cntSamples);
% calculate the strong classifier f(x) response.
for i=1:T
  predout = WaldBoostWeakLearnerClassfy(Samples,Y,Hypothesis(i,:));
  weakLearnerOutput(i,:) = predout;
end
AlphaTMatrix = repmat(AlphaT',[1 cntSamples]);
weakLearnerOutput = AlphaTMatrix.*weakLearnerOutput;
interOutput = sum(weakLearnerOutput,1);

%divding the response by setting thresA and thresB.
%compute frequency
inter_Freq = tabulate(interOutput);
positiveCols=find(Y==1);
negativeCols=find(Y==-1);
interOutput_true = interOutput(positiveCols);
interOutput_false = interOutput(negativeCols);  
inter_Freq_true = tabulate(interOutput_true);   % distribution of predict value of true sample
inter_Freq_false = tabulate(interOutput_false); % distribution of predict value of false sample

potential_thres = inter_Freq(:,1);  % all potential threshold for waldboost
TPcount = 0;TNcount = 0;FPcount = 0;FNcount = 0;Ncount = 0;Pcount = 0;
thresA = 0;thresB = 0;
al = 0.01;
be = 0.01;
A = (1-be)/al;  % initialize thres for SPRT
B = be/(1-al);
noupper_thres = false;
nolower_thres = false;
thresA = NaN;
thresB = NaN;
for i = 1:length(potential_thres)   % to find the lower threshold thresA
  col_idx_f = find(inter_Freq_false(:,1) == potential_thres(i));
  col_idx_t = find(inter_Freq_true(:,1) == potential_thres(i));
  if(~isempty(col_idx_f))
    FNcount = FNcount + inter_Freq_false(col_idx_f,2);
  end
  if(~isempty(col_idx_t))
    TNcount = TNcount + inter_Freq_true(col_idx_t,2);
  end
  if(FNcount < A * TNcount) % loop end when SPRT rule not meet
    if(i == 1)
      nolower_thres = true;
    else
      thresA = potential_thres(i-1) + eps;  % calculate the lower threshold
    end
    break;
  end
  Ncount = Ncount + inter_Freq(i,2);
  idx = find(interOutput < potential_thres(i)+eps);
  predictOutput(idx) = -1;
end

for i = length(potential_thres):-1:1  % to find the upper threshold thresA
  col_idx_f = find(inter_Freq_false(:,1) == potential_thres(i));
  col_idx_t = find(inter_Freq_true(:,1) == potential_thres(i));
  if(~isempty(col_idx_f))
    FPcount = FPcount + inter_Freq_false(col_idx_f,2);
  end
  if(~isempty(col_idx_t))
    TPcount = TPcount + inter_Freq_true(col_idx_t,2);
  end
  if(FPcount > B * TPcount) % loop end when SPRT rule not meet
    if(i == length(potential_thres))
      noupper_thres = true;
    else
      thresB = potential_thres(i+1) - eps;  % calculate the upper threshold
    end
    break;
  end
  Pcount = Pcount + inter_Freq(i,2);
  idx = find(interOutput > potential_thres(i)+eps);
  predictOutput(idx) = 1;
end

if(~ nolower_thres || ~ noupper_thres) % check if there are lower and upper threshold
else
  thresA = NaN;
  thresB = NaN;
end
%[Ncount Pcount thresA thresB] % debug info









