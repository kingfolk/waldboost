function [errorRate,overallErrorRate,TPRate,FPRate]=calPredictErrorRate(stdOutput,predictOutput)
%if(Hypothesis(T,4) == NaN)
%	ErrorRate = 1;
%	OverallErrorRate = 1;
%	TPrate = 1;
%	FPrate = 1;
%else
%  if(T~=1) 
%    idx = find(decision~=0);
%    predictOutput(idx) = decision(idx);
%  end
	TPSamples=((predictOutput+stdOutput)); 
	ErrorSamples=stdOutput-predictOutput;      
	%length(find(ErrorSamples~=0))		% debug info
	overallErrorRate=length(find(ErrorSamples~=0))/length(stdOutput);
	errorRate =length(find(TPSamples==0))/(length(stdOutput)-length(find(abs(TPSamples)==1)));
	%length(stdOutput)
	%length(find(abs(TPSamples)==1))

                                           % calculate error rate
	TPRate=length(find(TPSamples==2))/length(find(stdOutput==1));  
                                           % TP
	FPRate=length(find(ErrorSamples==-2))/length(find(stdOutput==-1)); 
%end


