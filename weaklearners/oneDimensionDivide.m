%
%
%						|
%	negative samples 	|	positive samples
%						|
%		smaller	 <- threshold -> larger  -> Bias = 1
%
function [Error,Thresh,Bias] = oneDimensionDivide(X, Y, weight)
iptcheckinput(X,{'numeric'},{'real','vector'}, mfilename,'X',1);
iptcheckinput(Y,{'numeric'},{'real','vector'}, mfilename,'Y',1);
cntSamples = length(X);
[X_sort idx_sort] = sort(X);
Y_sort = Y(idx_sort);
weight_sort = weight(idx_sort);
bestError = 1;
startError = sum(weight_sort(find(Y_sort == -1)));
errorrate = startError;
bestThresh = X_sort(1);
bestBias = 1;
for i = 1:cntSamples-1
	if(Y_sort(i) == -1)
		errorrate = errorrate - weight_sort(i);
	else
		errorrate = errorrate + weight_sort(i);
	end
	
	%YY = [zeros(1,i)-1 ones(1,cntSamples-i)];
	%errorrate = sum(weight_sort(Y_sort~=YY));
	%prediout = (X_sort>X_sort(i))*2-1;
	%errorrate = sum(weight_sort(Y_sort~=prediout'));
	if(errorrate > 0.5)
		tmperror = 1 - errorrate;
		tmpbias = -1;
	else
		tmperror = errorrate;
		tmpbias = 1;
	end
	if(tmperror < bestError)
		if(X_sort(i+1) == X_sort(i))
			continue;
		end
		bestError = tmperror;
		if(i ~= cntSamples)
			bestThresh = X_sort(i) + eps;
		else
			bestThresh = X_sort(i) + eps;
		end
		bestBias = tmpbias;
	end
end

% for i = 1:cntSamples
% 	tmpthresh = X(i);
% 	prediout = (X>tmpthresh)*2-1;
% 	tmperror = sum(weight(Y~=prediout'));
% 	if (tmperror > 0.5)
% 		err = 1 - tmperror;
% 		tmpbias = -1;
% 	else
% 		err = tmperror;
% 		tmpbias = 1;
% 	end

% 	if(err < bestError) 
% 		bestError = err;
% 		bestThresh = tmpthresh;
% 		bestBias = tmpbias;
% 	end
% end

Error = bestError;
Thresh = bestThresh;
Bias = bestBias;
% bestError;
% predictOutput=(Bias.*X>Bias*Thresh)*2 - 1;
% sum(weight(predictOutput~=Y_sort));
%debug info
% [Error Thresh Bias];

% posCount_below = 0; posCount_above = length(find(Y_sort == 1));
% negCount_below = 0; negCount_above = length(find(Y_sort == -1));
% bestError = 1;
% errorrate_PosBias = sum(weight_sort(find(Y_sort == -1)));
% errorrate_NegBias = sum(weight_sort(find(Y_sort == 1)));
% for i = 1:cntSamples
% 	if (Y_sort(i) == 1)
% 		posCount_below = posCount_below + 1;
% 		posCount_above = posCount_above - 1;
% 		errorrate_PosBias = errorrate_PosBias - weight_sort(i);
% 		errorrate_PosBias = errorrate_PosBias
% 	else
% 		negCount_below = negCount_below + 1;
% 		negCount_above = negCount_above - 1;
% 	end
% 	errorrate_PosBias = 


% end




