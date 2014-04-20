%Input
%	X 			cntSamples*cntPixels, cntPixels here is equal to 256
%	Y 			tags for each training samples
%				cntSamples*1
%	weight 		weight for each sample
%				cntSamples*1
%	Hypothesis	structure [thetaA thetaB weaklearnertype channel threshold bias [lengthofparam parameters]]
%
%Output
%	predictOutput	
%				the predict tag of one weak hypothesis
%

function [predictOutput]=WaldBoostWeakLearnerClassfy(X,Y,Hypothesis)

weaklearnertype = Hypothesis(3);
channelChoice = Hypothesis(4);
thresh = Hypothesis(5);
bias = Hypothesis(6);
lengthofparam = Hypothesis(7);
param = Hypothesis(7:7+lengthofparam); % [lengthofparam parameters]

X_onechannel = X(:, (channelChoice-1)*256+1:channelChoice*256);
cntSamples = size(X_onechannel,1);
weight = ones(1,cntSamples)/cntSamples;
switch weaklearnertype
	case 1
		[Error,Hypothesis,predictOutput] = applyWeaklearnerMean(X_onechannel, Y, weight, param, [thresh bias]);
	case 2
		[Error,Hypothesis,predictOutput] = applyWeaklearnerMeanRatio(X_onechannel, Y, weight, param, [thresh bias]);
	case 3
		[Error,Hypothesis,predictOutput] = applyWeaklearnerDescriptor(X_onechannel, Y, weight, param, [thresh bias]);
	case 4
		[Error,Hypothesis,predictOutput] = applyWeaklearnerOther(X_onechannel, Y, weight, param, [thresh bias]);
	otherwise
		error('Error when choosing weak learner.');
end
