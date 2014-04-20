
positiveX = rand([1 1000]) * 10;
negativeX = rand([1 1000]) * 10 -5;
X = [positiveX negativeX];
Y = [ones([1 1000]) ones([1 1000])-2];
weight = [ones([1 2000])]/2000.0;
cntSamples = length(X);

iteration=4;
sectNum=8;

u1 = mean(positiveX);
u2 = mean(negativeX);

maxFea = max(u1,u2);
minFea = min(u1,u2);

step=(maxFea-minFea)/(sectNum-1);
bestError=1;

tic;

for iter = 1:iteration
	for i = 1:sectNum
		thresh = minFea+(i-1)*step;
		h=(X<thresh)*2-1;
		p = 1;
		errorrate=sum(weight(find(h~=Y)));
		if(errorrate>0.5)
			errorrate = 1 - errorrate;
			p = -1;
		end
		if(errorrate<bestError)
			bestError = errorrate;
			bestThresh = thresh;
			bestBias = p;
		end
	end
	span = (maxFea-minFea)/8;
	maxFea=bestThresh+span;
	minFea=bestThresh-span;
	step=(maxFea-minFea)/(sectNum-1);
end

toc;

[bestError bestThresh bestBias]

tic;
[X_sort idx_sort] = sort(X);
Y_sort = Y(idx_sort);
bestError = cntSamples/2;
errorrate = bestError;
bestThresh = 0;
for i = 1:cntSamples
	if(Y_sort(i) == 1)
		errorrate = errorrate - 1;
	else
		errorrate = errorrate + 1;
	end
	if(errorrate < bestError)
		bestError = errorrate;
		bestThresh = (X_sort(i)+X_sort(i+1))/2;
	end

end
errorrate = cntSamples/2;
bestThresh = 0;
for i = 1:cntSamples
	if(Y_sort(i) == -1)
		errorrate = errorrate - 1;
	else
		errorrate = errorrate + 1;
	end
	if(errorrate < bestError)
		bestError = errorrate;
		bestThresh = (X_sort(i)+X_sort(i+1))/2;
	end

end
toc;
[bestError/2000 bestThresh]


tic;
bestError = cntSamples/2;
errorrate = bestError;
bestThresh = 0;
for i = 1:cntSamples
	if(Y_sort(i) == 1)
		errorrate = errorrate - 1;
	else
		errorrate = errorrate + 1;
	end
	if(errorrate > cntSamples/2)
		tmperror = cntSamples - errorrate;
		tmpbias = -1;
	else
		tmperror = errorrate;
		tmpbias = 1;
	end
	if(tmperror < bestError)
		bestError = tmperror;
		if(i ~= cntSamples)
			bestThresh = (X_sort(i)+X_sort(i+1))/2;
		else
			bestThresh = X_sort(i) + 1;
		end
		bestBias = tmpbias;
	end

end

Error = bestError/cntSamples;
Thresh = bestThresh;
Bias = bestBias;
toc;
[Error Thresh Bias]






