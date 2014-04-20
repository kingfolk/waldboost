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
%				if manually set, the format should be [lengthofparam parameters]
%				parameters will be format of [rectStartX rectStartY sideLength]
%	varargin	this variable is to set the [threshold bias] term when we know the hypothesis at testing phase.
%
%Output
%	Hypothesis	structure [threshold bias [lengthofparam parameters]],which is different form applyBestAdaBoostWeakLearner.m
%	Error 		the misclassified rate under this hypothesis
%
function [Error Hypothesis predictOutput] = applyWeaklearnerMean(X,Y,weight,param,varargin)
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
			[rectStartX] = randi([1 13]);
			[rectStartY] = randi([1 13]);
			sideLengthX = randi([4 12]);
			sideLengthY = randi([4 12]);
			largerStart = max([rectStartX rectStartY]);
			if(rectStartX + sideLengthX -1 <= widthPatch && ...
				rectStartY + sideLengthY -1 <= heightPatch)
				break;
			end
		end
	end
else
	%parameters are set by input
	rectStartX = param(2);rectStartY = param(3);sideLengthX = param(4);sideLengthY = param(5);
	largerStart = max([rectStartX rectStartY]);
	if(rectStartX + sideLengthX -1 > widthPatch || ...
		rectStartY + sideLengthY -1 > heightPatch)
		error('applyWeaklearnerMeanRatio: the rectangle size exceed the patch.');
	end
end


% calculate the score under current weak learner.
% first convert 2d coordinate to 1d.
%
%	 ----------------------------------> COL/ X
%	|
%	|	coor1d00			coor1d01
%	|	1111111111111111111111
%	|	2222222222222222222222
%	|	3333333333333333333333		-->  11..1122....44 = X(i,:)
%	|	4444444444444444444444
%	|	coor1d10			coor1d11
%	v
% ROW/Y
% rectStartX_1 = rectStartX-1;
% rectStartY_1 = rectStartY-1;
% coor1d00 = rectStartX_1 + (rectStartY_1-1)*widthPatch;
% coor1d01 = rectStartX + (rectStartY_1-1)*widthPatch + sideLength -1;
% coor1d10 = rectStartX_1 + (rectStartY-1 + sideLength-1)*widthPatch;
% coor1d11 = rectStartX + (rectStartY-1 + sideLength-1)*widthPatch + sideLength-1;

% % border condition
% comp11 = X(:,coor1d11);
% if(rectStartX==1) comp10 = 0;else comp10 = X(:,coor1d10); end
% if(rectStartY==1) comp01 = 0;else comp01 = X(:,coor1d01); end
% if(rectStartY==1||rectStartX==1) comp00 = 0;else comp00 = X(:,coor1d00); end
	
% scoreArr = comp00 + comp11 - comp10 - comp01;
scoreArr = computeIntegral(X,[rectStartX rectStartY sideLengthX sideLengthY],[heightPatch widthPatch]);

% this is only for testing phase.
if(nargin > 4)
	hyp = varargin{1};
	Thresh = hyp(1);
	Bias = hyp(2);
	Hypothesis = [Thresh Bias param];
	predictOutput=(Bias.*scoreArr'>Bias*Thresh)*2 - 1;
	Error = sum(weight(find(Y ~= predictOutput))); % wrong error.
	return;
end

% searching the best dividing plane.
[Error,Thresh,Bias] = oneDimensionDivide(scoreArr, Y, weight);
predictOutput=(Bias.*scoreArr'>Bias*Thresh)*2 - 1;

outputParam = [4 rectStartX rectStartY sideLengthX sideLengthY];
Hypothesis = [Thresh Bias outputParam];
% debug info
%scoreArr
%[coor1d00 coor1d01 coor1d10 coor1d11]
%[rectStartX rectStartY sideLength]
%[Error,Thresh,Bias]





