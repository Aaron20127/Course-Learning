% Auto-generated by cameraCalibrator app on 07-Oct-2018
%-------------------------------------------------------

% Define images to process
imageFileNames = {'picture\WIN_20181007_19_26_04_Pro.jpg',...
    'picture\WIN_20181007_19_26_23_Pro.jpg',...
    'picture\WIN_20181007_19_26_36_Pro.jpg',...
    'picture\WIN_20181007_19_26_48_Pro.jpg',...
    'picture\WIN_20181007_19_26_57_Pro.jpg',...
    'picture\WIN_20181007_19_27_02_Pro.jpg',...
    'picture\WIN_20181007_19_27_08_Pro.jpg',...
    'picture\WIN_20181007_19_27_13_Pro.jpg',...
    'picture\WIN_20181007_19_27_22_Pro.jpg',...
    'picture\WIN_20181007_19_27_30_Pro.jpg',...
    'picture\WIN_20181007_19_27_57_Pro.jpg',...
    'picture\WIN_20181007_19_28_10_Pro.jpg',...
    'picture\WIN_20181007_19_28_22_Pro.jpg',...
    'picture\WIN_20181007_19_28_25_Pro.jpg',...
    'picture\WIN_20181007_19_28_29_Pro.jpg',...
    'picture\WIN_20181007_19_28_43_Pro.jpg',...
    };
% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates of the corners of the squares
squareSize = 60;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'millimeters', ...
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

figure; 
subplot(1,2,1);imshow(originalImage),title('ԭʼͼƬ');
subplot(1,2,2);imshow(undistortedImage),title('�궨��ͼƬ');

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')