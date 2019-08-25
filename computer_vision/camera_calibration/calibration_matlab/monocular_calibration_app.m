% ------------------------------------------
% calibrate monocular images by matlab
% ------------------------------------------


% Define images to process
imageFileNames = {'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\1.jpg',...
    'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\11.jpg',...
    'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\12.jpg',...
    'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\14.jpg',...
    'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\2.jpg',...
    'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\4.jpg',...
    'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\6.jpg',...
    'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\9.jpg',...
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
