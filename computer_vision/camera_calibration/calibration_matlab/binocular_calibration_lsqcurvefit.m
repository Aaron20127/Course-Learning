
main()

function main()
% get images full name from dir
squareSize = 10;  % in units of 'millimeters'
leftFileDir =  'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\';
rightFileDir = 'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\right\';

[leftImagePoints, worldPoints, mrows, ncols] = getImageAndWorldPoints(leftFileDir, '*.jpg', squareSize);
[rightImagePoints, worldPoints, mrows, ncols] = getImageAndWorldPoints(rightFileDir, '*.jpg', squareSize);

% Calibrate the camera 1
[cameraParams1, imagesUsed1, estimationErrors1] = estimateCameraParameters(leftImagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% Calibrate the camera 2
[cameraParams2, imagesUsed2, estimationErrors2] = estimateCameraParameters(rightImagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);


% initialize
[x0, xdata, ydata] = initParams(cameraParams1, cameraParams2, worldPoints, leftImagePoints, rightImagePoints);

% LM algorithm
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
lb = [];
ub = [];    
options.MaxFunctionEvaluations = inf;
options.MaxIterations = inf;
options.FunctionTolerance = 1.0000e-6;
options.StepTolerance = 1.0000e-6;

[x,resnorm,residual,exitflag,output] = lsqcurvefit(@projectStereo, x0, xdata, ydata, lb, ub, options)

% show params
x
resnorm
meanError=mean(abs(residual))

Rv = x(end-5:end-3);
R = vision.internal.calibration.rodriguesVectorToMatrix(Rv)
T = x(end-2:end)

intrinsic1 = x(end-17 : end-12);
intrinsic2 = x(end-11 : end-6);

Ml = [intrinsic1(1), 0, intrinsic1(3); 0, intrinsic1(2), intrinsic1(4); 0, 0, 1];
Mr = [intrinsic2(1), 0, intrinsic2(3); 0, intrinsic2(2), intrinsic2(4); 0, 0, 1];

Tx = [0,-T(3),T(2);T(3),0,-T(1);-T(2),T(1),0];
E = R * Tx
F = inv(Mr') * R * Tx * inv(Ml)

end


% initialize x0, xdata, ydata, which are used for iteration.
%----------------------------------------------------------------------
function [x0, xdata, ydata] = initParams(cameraParams1, cameraParams2, worldPoints, leftImagePoints, rightImagePoints)
    % initialize x0 = [r1, t1, r2, t2, ... fx1, fy2, ... , fx1, fy2, ..., Rv, T]
    rvecs1 = cameraParams1.RotationVectors;
    tvecs1 = cameraParams1.TranslationVectors;
    intrinsic1 = [cameraParams1.FocalLength, ...
                cameraParams1.PrincipalPoint, ...
                cameraParams1.RadialDistortion]';

    rvecs2 = cameraParams2.RotationVectors;
    tvecs2 = cameraParams2.TranslationVectors;
    intrinsic2 = [cameraParams2.FocalLength, ...
                cameraParams2.PrincipalPoint, ...
                cameraParams2.RadialDistortion]';

    numPattens = size(rvecs1, 1);
    % init R, T
    [Rv, T] = initRT(rvecs1, rvecs2, tvecs1, tvecs2);
    % init r, v
    rt = reshape([rvecs1, tvecs1]', 6*numPattens, 1);
    % [r1; v1; r2; v2; ... fx; ... ; fx; ...; Rv; T]
    x0 = [rt; intrinsic1; intrinsic2; Rv; T];

    % initialize xdata
    xdata = worldPoints(:);

    % initialize ydata, [u(img1);v(img1);u(img2);v(img2); ...; u(imgn);v(imgn);]
    ydata = [];
    for i = 1:numPattens
        temp = leftImagePoints(:,:,i);
        ydata = [ydata; temp(:)];
    end

    for i = 1:numPattens
        temp = rightImagePoints(:,:,i);
        ydata = [ydata; temp(:)];
    end
end

% initialize R,T, which is coordinates frams transform matrix beteen camera one two camera two
%----------------------------------------------------------------------
function [Rv, T] = initRT(rvecs1, rvecs2, tvecs1, tvecs2)

    numPattens = size(rvecs1, 1);
    rm = [];
    tm = [];

    for i = 1:numPattens
        R1 = vision.internal.calibration.rodriguesVectorToMatrix(rvecs1(i,:));
        R2 = vision.internal.calibration.rodriguesVectorToMatrix(rvecs2(i,:));
        t1 = tvecs1(i,:)';
        t2 = tvecs2(i,:)';

        R = R2*R1';
        Rv= vision.internal.calibration.rodriguesMatrixToVector(R);
        T = t2 - R*t1;

        rm = [rm;Rv'];
        tm = [tm;T'];
    end
    
    Rv = median(rm)';
    T = median(tm)';
end

% project for camera one and camera two
% reprojection func for LM algorithm, only support one image 
% numerical differentiation using finite difference approximations
% address: https://toutiao.io/posts/499106/app_preview
%----------------------------------------------------------------------
function value=projectStereo(x, xdata) 
    totalparams = size(x,1);
    rt1 = x(1:totalparams-18);
    intrinsic1 = x(end-17 : end-12);
    intrinsic2 = x(end-11 : end-6);
    Rv = x(end-5 : end-3);
    R = vision.internal.calibration.rodriguesVectorToMatrix(Rv);
    T = x(end-2 : end);
    
    % project camera 1
    x0 = [rt1; intrinsic1];
    f1 = projectSingleCamera(x0, xdata);
    
    % project camera 2
    rt2 = [];
    rt1 = reshape(rt1, 6, size(rt1, 1)/6);
    numPattens = size(rt1, 2);
   
    for i = 1:numPattens
       r1 = rt1(1:3,i);
       t1 = rt1(4:6,i);
       R1 = vision.internal.calibration.rodriguesVectorToMatrix(r1);

       R2 = R1*R;
       t2 = T + R*t1;
       r2 = vision.internal.calibration.rodriguesMatrixToVector(R2);

       rt2 = [rt2; r2; t2];
    end

    x1 = [rt2; intrinsic2];
    f2 = projectSingleCamera(x1, xdata);

    % return
    value = [f1; f2];
end

% single camera projection
%----------------------------------------------------------------------
function value=projectSingleCamera(x, xdata) 
    totalPara = size(x,1);
    numIntrinsic = 6; % fx, fy, cx, cy, k1, k2,
    numExtrinsic = totalPara - numIntrinsic;
    numImages = numExtrinsic / 6;

    intrinsics = x(numExtrinsic+1 : totalPara);
    extrinsics = reshape(x(1:numExtrinsic), 6, numImages);

    [row, col] = size(xdata);
    value = zeros(numImages*row*col, 1);
    
    %  Worlds coordinates
    xw = xdata(1:row/2);
    yw = xdata(row/2+1:row);

    for i = 1:numImages
        tv = extrinsics(:,i);
        r  = vision.internal.calibration.rodriguesVectorToMatrix(tv(1:3));

        r11 = r(1,1);
        r12 = r(1,2);
        r13 = r(1,3);
        r21 = r(2,1);
        r22 = r(2,2);
        r23 = r(2,3);
        r31 = r(3,1);
        r32 = r(3,2);
        r33 = r(3,3);
        t1  = tv(4);
        t2  = tv(5);
        t3  = tv(6);
        fx  = intrinsics(1);
        fy  = intrinsics(2);
        cx  = intrinsics(3);
        cy  = intrinsics(4);
        k1  = intrinsics(5);
        k2  = intrinsics(6);

        %   Normalized coordinates
        xn = (r11 * xw + r12 * yw + t1) ./ (r31 * xw + r32 * yw + t3);
        yn = (r21 * xw + r22 * yw + t2) ./ (r31 * xw + r32 * yw + t3);

        %   r^2, r^4
        r2 = xn .* xn + yn .* yn;
        r4 = r2 .* r2;
        r6 = r4 .* r2;

        %   reprojection
        value(((i-1)*row*col+1): i*row*col, :) = ...
            [fx * (xn .* (1 + k1*r2 + k2*r4)) + cx; ...
             fy * (yn .* (1 + k1*r2 + k2*r4)) + cy];
    end

end

%--------------------------------------------------------------------------
function filename = getFileNameFromDir(filedir, patten)
% get file full name from file dir patten(such as '/E:/*.jpg')
imageStruct = dir([filedir patten]);
numImage = size(imageStruct,1);

filename = cell(numImage,1);

for i=1:numImage
    filename(i)= {strcat(filedir, imageStruct(i).name)};
end
end


%--------------------------------------------------------------------------
function [imagePoints, worldPoints, mrows, ncols] = getImageAndWorldPoints(dir, patten, squareSize)
% @dir: image dir
% @patten: file type, such as *.jpg
% @squareSize: calibration patten grid size

imageFileNames = getFileNameFromDir(dir, patten);

% Read one of the images from the first stereo pair
I1 = imread(imageFileNames{1});
[mrows, ncols, ~] = size(I1);

% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates of the corners of the squares
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

end

