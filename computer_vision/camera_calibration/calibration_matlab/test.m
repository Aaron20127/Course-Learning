% xdata = [1,2; 4,5; 4,6]

% % fun = @(x,xdata) [x(1)*xdata(:,1).*xdata(:,1), x(2)*xdata(:,2).*xdata(:,2)]

% ydata = fun([2,3;1,2], xdata)

% x0 = [4,5;2,3]
% [x,resnorm,residual,exitflag,output] = lsqcurvefit(@fun,x0,xdata,ydata)




% worldsPoints = [];
% worldsPoints = cameraParams.WorldPoints;
% 
% % for i = 1:8
% %     worldsPoints = [worldsPoints; cameraParams.WorldPoints];
% % end
% 
% imagesPoints = [];
% imagesPoints = imagePoints(:,:,8);
% % for i = 1:8
% %     imagesPoints = [imagesPoints; imagePoints(:,:,i)];
% % end
% 
% x = [fx, fy, cx, cy, k1, k2, r11, r21, r31, r12, r22, r32, t1, t2, t3]
% fun @(x,xdata) [x(1)*xdata(:,1).*xdata(:,1), x(2)*xdata(:,2).*xdata(:,2)]
% 
% xdata = worldsPoints;
% ydata = imagesPoints;
% x0 = [500, 500, 300, 200, -0.2, 0.1, 0.14, -0.5, -0.5, 09, 0.2, 0.1, 20, -20, 100];
% x0 = [453, 473, 162, 221, 221, ]
% 
% options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
% lb = [];
% ub = [];
% options.MaxFunctionEvaluations = inf;
% options.MaxIterations = inf;
% options.FunctionTolerance = 1.0000e-16;
% options.StepTolerance = 1.0000e-6;
% 
% [x,resnorm,residual,exitflag,output] = lsqcurvefit(@reProjection,x0,xdata,ydata,lb, ub,options);
% x,resnorm,,exitflag,output
% 
% function value=reProjection(x,xdata) 
%     
%     fx = x(1);
%     fy = x(2);
%     cx = x(3);
%     cy = x(4);
%     k1 = x(5);
%     k2 = x(6);
%     r11 = x(7);
%     r21 = x(8);
%     r31 = x(9);
%     r12 = x(10);
%     r22 = x(11);
%     r32 = x(12);
%     t1 = x(13);
%     t2 = x(14);
%     t3 = x(15);
% 
%     Worlds coordinates
%     xw = xdata(:,1);
%     yw = xdata(:,2);
% 
%     Normalized coordinates
%     xn = (r11 * xw + r12 * yw + t1) ./ (r31 * xw + r32 * yw + t3);
%     yn = (r21 * xw + r22 * yw + t2) ./ (r31 * xw + r32 * yw + t3);
% 
% %      r^2, r^4
%     r2 = xn .* xn + yn .* yn;
%     r4 = r2 .* r2;
% 
%     reprojection
%     value = [fx * xn .* (1 + k1*r2 + k2*r4) + cx, fy * yn .* (1 + k1*r2 + k2*r4) + cy];
% end








% Define images to process
imageFileNames = {'F:\VM\VM share\ubuntu-18-04\data\left\1.jpg',...
    'F:\VM\VM share\ubuntu-18-04\data\left\2.jpg',...
    'F:\VM\VM share\ubuntu-18-04\data\left\4.jpg',...
    'F:\VM\VM share\ubuntu-18-04\data\left\6.jpg',...
    'F:\VM\VM share\ubuntu-18-04\data\left\9.jpg',...
    'F:\VM\VM share\ubuntu-18-04\data\left\11.jpg',...
    'F:\VM\VM share\ubuntu-18-04\data\left\12.jpg',...
    'F:\VM\VM share\ubuntu-18-04\data\left\14.jpg',...
    };
% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates of the corners of the squares
squareSize = 10;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')







% function value = fun(x,xdata)
%     value = [x(1,1)*xdata(:,1).*xdata(:,1), x(1,2)*xdata(:,2).*xdata(:,2)]
% end

