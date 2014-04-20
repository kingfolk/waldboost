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
%				parameters will be format of Param = [rectStartX rectStartY sideLengthX sideLengthY bestBin]
%	varargin	this variable is to set the [threshold bias] term when we know the hypothesis at testing phase.
%
%Output
%	Hypothesis	structure [threshold bias [lengthofparam parameters]],which is different form applyBestAdaBoostWeakLearner.m
%	Error 		the misclassified rate under this hypothesis
%

function [Error Hypothesis predictOutput] = applyWeaklearnerDescriptor(X,Y,weight,param,varargin)
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
			[rectStartX] = randi([2 13]);
			[rectStartY] = randi([2 13]);
			sideLengthX = randi([4 12]);
			sideLengthY = randi([4 12]);
			% the condition without -1 because we need to remain the border
			if(rectStartX + sideLengthX  <= widthPatch && ...
				rectStartY + sideLengthY  <= heightPatch)
				break;
			end
		end
	end
else
	%parameters are set by input
	rectStartX = param(2);rectStartY = param(3);sideLengthX = param(4);sideLengthY = param(5);
	if(rectStartX + sideLengthX > widthPatch || ...
		rectStartY + sideLengthY > heightPatch)
		error('applyWeaklearnerMeanRatio: the rectangle size exceed the patch.');
	end
end

%									X
%	--------------------------------->
%	|
%	|		coA->	XXX
%	|		 		 O
%	|		coB->	XXX
%	|
%	|		coC->	X X
%	|				XOX
%	|				X X <-coD
%	|
% Y	V
%

Gmag = zeros(sideLengthY,sideLengthX,cntSamples);
Gdir = zeros(sideLengthY,sideLengthX,cntSamples);
for i = 1:sideLengthY
	for j = 1:sideLengthX
		coA = [rectStartX+j-2 rectStartY+i-2 3 1];
		coB = [rectStartX+j-2 rectStartY+i 3 1];
		coC = [rectStartX+j-2 rectStartY+i-2 1 3];
		coD = [rectStartX+j rectStartY+i-2 1 3];
		compA = computeIntegral(X,coA,[heightPatch widthPatch]);
		compB = computeIntegral(X,coB,[heightPatch widthPatch]);
		compC = computeIntegral(X,coC,[heightPatch widthPatch]);
		compD = computeIntegral(X,coD,[heightPatch widthPatch]);
		%Prewitt operator
		Gy = compA - compB;
		Gx = compD - compC;
		Gmag(i,j,:) = sqrt(Gy.*Gy + Gx.*Gx);
		Gdir(i,j,:) = atan2(Gy,Gx);
	end
end

Gbin = zeros(8,cntSamples);
for bin = 1:8
	ang = (bin-5)/4*pi;
	for i = 1:sideLengthY
		for j = 1:sideLengthX
			idx = find(Gdir(i,j,:)>=ang & Gdir(i,j,:)<ang+pi/4);
			%[size(Gbin(bin,idx)) size(Gmag(i,j,idx))]
			Gmag2D = reshape(Gmag(i,j,idx),[1 length(idx)]);
			Gbin(bin,idx) = Gbin(bin,idx) + Gmag2D;
		end
	end
end

kernel = fspecial('gaussian',[1 9],3);
kernel = kernel(3:7); %only keep the center weight
kernelMatrix = repmat(kernel',[1 cntSamples]); % 5 * cntSamples matrix
if(nargin > 4)
	hyp = varargin{1};
	Thresh = hyp(1);
	Bias = hyp(2);
	bestBin = param(6);
	Hypothesis = [Thresh Bias param];

	binRange = [bestBin-2:bestBin+2];
	binRange(find(binRange<1)) = binRange(find(binRange<1)) + 8;
	binRange(find(binRange>8)) = binRange(find(binRange>8)) - 8;
	binScoreMt = kernelMatrix .* Gbin(binRange,:);
	binScore = sum(binScoreMt,1);
	predictOutput=(Bias.*binScore>Bias*Thresh)*2 - 1;
	Error = sum(weight(find(Y ~= predictOutput))); % error for current weak hypothesis
	return;
end


bestError = 1;
for bin = 1:8
	binRange = [bin-2:bin+2];
	binRange(find(binRange<1)) = binRange(find(binRange<1)) + 8;
	binRange(find(binRange>8)) = binRange(find(binRange>8)) - 8;
	binScoreMt = kernelMatrix .* Gbin(binRange,:);
	binScore = sum(binScoreMt,1);
	[tmpError,tmpThresh,tmpBias] = oneDimensionDivide(binScore, Y, weight);
	if(tmpError < bestError)
		bestError = tmpError;
		bestThresh = tmpThresh;
		bestBias = tmpBias;
		bestBin = bin;
		bestBinScore = binScore;
	end
end
Error = bestError;
Thresh = bestThresh;
Bias = bestBias;
predictOutput=(Bias.*bestBinScore>Bias*Thresh)*2 - 1;
outputParam = [5 rectStartX rectStartY sideLengthX sideLengthY bestBin];
Hypothesis = [Thresh Bias outputParam];
% if(Bias == -1)
% 	[1 length(find(bestBinScore(Y==1)))]
% 	[-1 length(find(bestBinScore(Y==-1)))]
% 	tabulate(bestBinScore(find(Y==1)))
% 	tabulate(bestBinScore(find(Y==-1)))
% end


