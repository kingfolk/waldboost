%This function try mean color weak learner over the input samples.
%	You can either use random parameter or not and get a result hypothesis
%	with a Error rate.
%
%Input
%	X 			cntSamples*cntPixels, cntPixels here is equal to 256
%	Y 			tags for each training samples
%				1*cntSamples
%	weight 		weight for each sample
%				1*cntSamples
%	param 		the parameter of the weak learner.
%				if randomly set, the input value should be 'random'
%				if manually set, the format should be  [lengthofparam parameters]
%				lengthofparam will be equal to 8
%				parameters will be format of [rect1StartX rect1StartY side1LengthX side1LengthY rect2StartX rect2StartY side2LengthX side2LengthY]
%	varargin	this variable is to set the [threshold bias] term when we know the hypothesis at testing phase.
%
%Output
%	Hypothesis	structure [threshold bias [lengthofparam parameters]],which is different form applyBestAdaBoostWeakLearner.m
%	Error 		the misclassified rate under this hypothesis
%
function [Error Hypothesis predictOutput] = applyWeaklearnerMeanRatio(X,Y,weight,param,varargin)
error(nargchk(4,5,nargin));
iptcheckinput(X,{'numeric'},{'2d','real','nonsparse'}, mfilename,'X',1);
iptcheckinput(Y,{'logical','numeric'},{'vector','nonempty','integer'},mfilename, 'Y', 2);
iptcheckinput(weight,{'numeric'},{'vector','nonempty','real'},mfilename, 'weight', 3);
iptcheckinput(param,{'char','numeric'},{'nonempty'},mfilename,'param',4);

[cntSamples cntPixels] = size(X);
widthPatch = 16;
heightPatch = 16;
if(ischar(param))
	if(strcmp(param,'random'))
		%random parameter generation
		while(1)
			[rect1StartX] = randi([1 13]);
			[rect1StartY] = randi([1 13]);
			side1LengthX = randi([4 12]);
			side1LengthY = randi([4 12]);

			if(rect1StartX + side1LengthX -1 <= widthPatch && ...
				rect1StartY + side1LengthY -1 <= heightPatch)
				break;
			end
		end
		while(1)
			[rect2StartX] = randi([1 13]);
			[rect2StartY] = randi([1 13]);
			side2LengthX = randi([4 12]);
			side2LengthY = randi([4 12]);

			if(rect2StartX + side2LengthX -1 <= widthPatch && ...
				rect2StartY + side2LengthY -1 <= heightPatch)
				break;
			end
		end

	end
else
	%parameters are set by input
	rect1StartX = param(2);rect1StartY = param(3);side1LengthX = param(4);side1LengthY = param(5);
	rect2StartX = param(6);rect2StartY = param(7);side2LengthX = param(8);side2LengthY = param(9);
	if(rect1StartX + side1LengthX -1 > widthPatch || ...
		rect1StartY + side1LengthY -1 > heightPatch || ...
		rect2StartX + side2LengthX -1 > widthPatch || ...
		rect2StartY + side2LengthY -1 > heightPatch)
		error('applyWeaklearnerMean: the rectangle size exceed the patch.');
	end
end


scoreArrRect1 = computeIntegral(X,[rect1StartX rect1StartY side1LengthX side1LengthY],[heightPatch widthPatch]);
scoreArrRect2 = computeIntegral(X,[rect2StartX rect2StartY side2LengthX side2LengthY],[heightPatch widthPatch]);
ratioArrRect = scoreArrRect1 ./ scoreArrRect2;

% this is only for testing phase.
if(nargin > 4)
	hyp = varargin{1};
	Thresh = hyp(1);
	Bias = hyp(2);
	Hypothesis = [Thresh Bias param];
	predictOutput=(Bias.*ratioArrRect'>Bias*Thresh)*2 - 1;
	Error = sum(weight(find(Y ~= predictOutput))); % wrong error.
	return;
end

% searching the best dividing plane.
[Error,Thresh,Bias] = oneDimensionDivide(ratioArrRect, Y, weight);
predictOutput=(Bias.*ratioArrRect'>Bias*Thresh)*2 - 1;

outputParam = [8 rect1StartX rect1StartY side1LengthX side1LengthY rect2StartX rect2StartY side2LengthX side2LengthY];
Hypothesis = [Thresh Bias outputParam];
% debug info
%scoreArr
%[coor1d00 coor1d01 coor1d10 coor1d11]
%[rectStartX rectStartY sideLength]
%[Error,Thresh,Bias]





