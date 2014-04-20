% Find the best hypothesis from a predefined hypothese set
%	The searching space only contain the subspace of all possible weak learners. 
%	We only choose the following weak learners from rect size 4 to 8.
%	A 		B 		C 		D
%	xxxx	xxoo	xxoo	ooxx
%	xxxx	xxoo	xxoo	ooxx
%	oooo	xxoo	ooxx	xxoo
%	oooo	xxoo	ooxx	xxoo
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


function [bestHypothesis bestPreictOutput] = BestAdaBoostWeakLearnerManual(X, Y, weight)

coor = [0 0 4 8 4 0 4 8;
		0 0 8 4 0 4 8 4;
		0 0 4 4 4 4 4 4;
		0 4 4 4 4 0 4 4];

bError = 1;
% a square rectangle 8*8
for k = 1:4 % for the 4 types of weak learner
	for i = 1:2:9
		for j = 1:2:9
			for chan = 1:3 % for 3 channels
				X_onechannel = X(:, (chan-1)*256+1:chan*256);
				coorA = coor(k,:)+[i j 0 0 i j 0 0]; %coor = [rectStartX rectStartY sideLengthX sideLengthY ....]
				[tmpError,tmpHypothesis,tmppredOutput] = applyWeaklearnerMeanRatio(X_onechannel, Y, weight,[length(coorA) coorA]);
				if(tmpError < bError)
					bError = tmpError;
					bHypothesis = tmpHypothesis;
					bpredOut = tmppredOutput;
					bchannel = chan;
				end
			end
		end
	end
end
bestHypothesis(1:2) = [0 0];				% thresA and thresB to be decided
bestHypothesis(3) = 2;
bestHypothesis(4) = bchannel;
hlength = length(bHypothesis);
bestHypothesis(5:4+hlength) = bHypothesis;
bestPreictOutput = bpredOut;
bError %debug info





