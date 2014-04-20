%function [predictOutput,ErrorRate,OverallErrorRate,TPrate,FPrate,thresA,thresB]=WaldBoostClassfy(Samples,Y,Hypothesis,AlphaT,T,varargin)
function [predictOutput,thresA,thresB]=WaldBoostCl(Samples,Y,Hypothesis,AlphaT,T,varargin)
error(nargchk(5,6,nargin));        % check input
iptcheckinput(Samples,{'numeric'},{'2d','real','nonsparse'}, mfilename,'Samples',1);
iptcheckinput(Hypothesis,{'numeric'},{'2d','real','nonsparse'}, mfilename,'Hypothesis',2);
iptcheckinput(AlphaT,{'numeric'},{'row','nonempty','real'},mfilename, 'AlphaT',3);

cntSamples=size(Samples,1);        % number of samples
boostthresh=false;                   % AdaBoost thres for weak learner
if( nargin>5 )                     % to set the thres
    boostthresh=varargin{1};
end
iptcheckinput(T,{'numeric'},{'row','nonempty','integer',},mfilename, 'T',4);
if( length(T) > 1 )              % not a vector
    error(['T should be a number.']);
end
iptcheckinput(boostthresh,{'numeric','logical'},{'row','nonempty','real'},mfilename, 'boostthresh',5);
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


%new procedure to find thresA and thresB.
[interOutput_sort idx_sort] = sort(interOutput);
Y_sort = Y(idx_sort);
negCount = length(find(Y_sort<0));
posCount = length(find(Y_sort>0));
negCount_below = 0;
negCount_above = negCount;
posCount_below = 0;
posCount_above = posCount;

thresA = 0;thresB = 0;
al = 0.01;
be = 0.01;
A = (1-be)/al;  % initialize thres for SPRT
B = be/(1-al);
noupper_thres = false;
nolower_thres = false;
thresA = NaN;
thresB = NaN;

for i = 1:cntSamples
	if(Y_sort(i) == -1)
		negCount_below = negCount_below + 1;
		negCount_above = negCount_above - 1;
	else
		posCount_below = posCount_below + 1;
		posCount_above = posCount_above - 1;
	end
	if(negCount_below > A * posCount_below)
		thresA = interOutput_sort(i)-eps;
	end
	if(negCount_above < B * posCount_above)
		thresB = interOutput_sort(i)+ eps;
		break;
	end
end

if(boostthresh) % last round
	predictOutput(interOutput<0) = -1;
  	predictOutput(interOutput>=0) = 1;
end

predictOutput(find(interOutput<thresA)) = -1;
predictOutput(find(interOutput>thresB)) = 1;

[thresA thresB] % debug info









