% Find the best hypothesis from a size of cascadeNum weak learners
%	Here we define different type of weak learners, which would be thousands of.
%	And the searching space only contain the subspace of the whole set. It is done
%	Manually selection or Random selection on type and parameters of weak learners.
%
% Input:
%	X 			integral patch 
%               cntSamples*256 matrix from cntSamples*16*16
%	Y 			tags for each training samples
%				1*cntSamples
%	weight 		weight for each sample
%				cntSamples*1
%	cascadeNum	the total searching steps
%
% Output
%	bestHypothesis
%				with structure [thetaA thetaB weaklearnertype channel threshold bias [lengthofparam parameters]]
%				thetaA and thetaB will be calculated in main training loop after. 
%				Here we just set zero for both of them.
%				The resting parameters are to be set here.
%	bestPreictOutput
%				The predict result by bestHypothesis
%
function [bestHypothesis bestPreictOutput] = BestAdaBoostWeakLearner(X, Y, weight, cascadeNum)
iptcheckinput(X,{'numeric'},{'2d','real','nonsparse'}, mfilename,'X',1);
iptcheckinput(Y,{'logical','numeric'},{'vector','nonempty','integer'},mfilename, 'Y', 2);
iptcheckinput(weight,{'numeric'},{'vector','nonempty','real'},mfilename, 'weight', 3);
iptcheckinput(cascadeNum,{'numeric'},{'integer'},mfilename, 'cascadeNum', 4);


minError = 1;
weakChoice = randi([2,3]);	% 1. choose over 4 types of weaker learners
%weakChoice = 3;			% 2. mannually set a type
if(weakChoice == 2)
	cascadeNum = 500;
else
	cascadeNum = 100;
end

for j=1:cascadeNum         % Over all feature to find best learner
	channelChoice = randi(3);
	X_onechannel = X(:, (channelChoice-1)*256+1:channelChoice*256);
	switch weakChoice
		case 1 				% 1. mean color of a random rectangle within a patch
			[tempError,Hypothesis,predout] = applyWeaklearnerMean(X_onechannel, Y, weight, 'random');
		case 2
			[tempError,Hypothesis,predout] = applyWeaklearnerMeanRatio(X_onechannel, Y, weight, 'random');
		case 3
			[tempError,Hypothesis,predout] = applyWeaklearnerDescriptor(X_onechannel, Y, weight, 'random');
		case 4
			[tempError,Hypothesis,predout] = applyWeaklearnerOther(X_onechannel, Y, weight, 'random');
		otherwise
			error('Error when choosing weak learner.');
	end
    if(tempError<minError)                       
        minError=tempError;                      	% t iteration
        bestHypothesis(1:2) = [0 0];				% thresA and thresB to be decided
        bestHypothesis(3) = weakChoice;
        bestHypothesis(4) = channelChoice;
        hlenght = length(Hypothesis);
        bestHypothesis(5:4+hlenght) = Hypothesis;	% set the parameter
        bestPreictOutput = predout;
    end
end
minError
bestHypothesis(3:6)
sum(weight(find(bestPreictOutput~=Y)))

