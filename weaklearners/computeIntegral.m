% given the required integral image and to-be-computed integral patch coordinate
% this function give a output integral value.
%
% Input
%	image	the integral image.
%			default to be 2d but also accept 1d array with a dim input in varagin
%			(if 1d array) cntSamples*cntPixels
%			(if 2d patch) height*width
%	coor 	the to-be-compute integral patch
%			[startX startY lengthX lengthY]
%	dim		(if set) [height width]		
%
% Output
%	integralOut 	the result
%			(if 1d array) 1*cntSamples
%			(if 2d patch) 1 value
%
%
function integralOut = computeIntegral(image,coor,varargin)
error(nargchk(2,3,nargin)); % 必须输入3-4个参数,否则中止程序
iptcheckinput(image,{'numeric'},{'real'}, mfilename,'image',1);
iptcheckinput(coor,{'numeric'},{'real','vector'}, mfilename,'coor',1);

rectStartX = coor(1);rectStartY = coor(2);lengthX = coor(3);lengthY = coor(4);
rectStartX_1 = rectStartX-1;rectStartY_1 = rectStartY-1;
if( nargin==2)
	if(rectStartX==1) comp10 = 0;else comp10 = image(rectStartX_1,rectStartY+lengthY-1); end
	if(rectStartY==1) comp01 = 0;else comp01 = image(rectStartX+lengthX-1,rectStartY_1); end
	if(rectStartY==1||rectStartX==1) comp00 = 0;else comp00 = image(rectStartX_1,rectStartY_1); end
	comp11 = image(rectStartX+lengthX-1,rectStartY+lengthY-1);
	integralOut = comp00 + comp11 - comp10 - comp01;

elseif( nargin>2 )
	dim = varargin{1};
	widthPatch = dim(2);

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
	coor1d00 = rectStartX_1 + (rectStartY_1-1)*widthPatch;
	coor1d01 = rectStartX + (rectStartY_1-1)*widthPatch + lengthX -1;
	coor1d10 = rectStartX_1 + (rectStartY-1 + lengthY-1)*widthPatch;
	coor1d11 = rectStartX + (rectStartY-1 + lengthY-1)*widthPatch + lengthX-1;

	% border condition
	comp11 = image(:,coor1d11);
	if(rectStartX==1) comp10 = 0;else comp10 = image(:,coor1d10); end
	if(rectStartY==1) comp01 = 0;else comp01 = image(:,coor1d01); end
	if(rectStartY==1||rectStartX==1) comp00 = 0;else comp00 = image(:,coor1d00); end
		
	integralOut = comp00 + comp11 - comp10 - comp01;
end



