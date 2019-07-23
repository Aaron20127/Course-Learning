


% get images full name from dir
squareSize = 10;  % in units of 'millimeters'
fileDir = 'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\binocular\left\'
imageFileNames = getFileNameFromDir(fileDir, '*.jpg')

% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates of the corners of the squares
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% get multiple homographies, init guess value 
[H, validIdx] = computeHomographies(imagePoints, worldPoints);
x0 = initGuessFromZhang(H);

% init xdata, ydata
imagesPoints = [];
for i = 1:size(imagePoints,3)
    imagesPoints = [imagesPoints; imagePoints(:,:,i)];
end
xdata = double(worldPoints);
ydata = double(imagesPoints);


% LM algorithm
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
lb = [];
ub = [];    
options.MaxFunctionEvaluations = inf;
options.MaxIterations = inf;
options.FunctionTolerance = 1.0000e-26;
options.StepTolerance = 1.0000e-16;

[x,resnorm,residual,exitflag,output] = lsqcurvefit(@reProjection,x0,xdata,ydata,lb, ub,options);
x,resnorm,residual,exitflag,output



% reprojection func for LM algorithm, only support one image 
%----------------------------------------------------------------------
function value=reProjection(x, xdata) 
    totalPara = size(x,2);
    numIntrinsic = 9;
    numExtrinsic = totalPara - numIntrinsic;
    numImages = numExtrinsic / 6;

    intrinsics = x(numExtrinsic+1 : totalPara);
    extrinsics = reshape(x(1:numExtrinsic), 6, numImages);

    [row, col] = size(xdata);
    value = zeros(numImages*row, col);

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
        k1  = intrinsics(1);
        k2  = intrinsics(2);
        k3  = intrinsics(3);
        p1  = intrinsics(4);
        p2  = intrinsics(5);
        fx  = intrinsics(6);
        fy  = intrinsics(7);
        cx  = intrinsics(8);
        cy  = intrinsics(9);

        %   Worlds coordinates
        xw = xdata(:,1);
        yw = xdata(:,2);

        %   Normalized coordinates
        xn = (r11 * xw + r12 * yw + t1) ./ (r31 * xw + r32 * yw + t3);
        yn = (r21 * xw + r22 * yw + t2) ./ (r31 * xw + r32 * yw + t3);

        %   r^2, r^4
        r2 = xn .* xn + yn .* yn;
        r4 = r2 .* r2;
        r6 = r4 .* r2;

        %   reprojection
        value(((i-1)*row+1): i*row, :) = ...
            [fx * (xn .* (1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*xn.*yn + p2*(r2 + 2*xn.*xn)) + cx, ...
             fy * (yn .* (1 + k1*r2 + k2*r4 + k3*r6) + 2*p2*xn.*yn + p1*(r2 + 2*yn.*yn)) + cy];
    end

end


%------------------------------------------------------------------------------------------------------
function x = initGuessFromZhang(homographies)
% x = [ rv1, rv2, rv3, tv1, tv2, tv3, ...
%       rv1, rv2, rv3, tv1, tv2, tv3, ...
%       rv1, rv2, rv3, tv1, tv2, tv3, ...
%       ...
%       rv1, rv2, rv3, tv1, tv2, tv3, ...
%       k1,  k2,  k3,  p1,   p2, fx, fy, cx, cy]
%
%       total = 6 * n + 9 

% init guess data as same as my python calibration init, use methods of Zhang
H = homographies;
V = computeV(H);
B = computeB(V);
A = computeIntrinsics(B);
[rvecs, tvecs] = computeExtrinsics(A, H);

% Combined into initial parameters
numImages = size(H,3);
numInternalPara = 9;
numExternalPara = 6*numImages;
x = ones(1, numInternalPara + numExternalPara);

tv = [rvecs; tvecs];
x(1:numExternalPara) = reshape(tv, 1, numExternalPara);
x(numExternalPara+1 : numExternalPara+numInternalPara) = ...
    [1, 1, 1, 1, 1, A(1,1), A(2,2), A(1,3), A(2,3)];

end


%--------------------------------------------------------------------------
function [homographies, validIdx] = computeHomographies(points, worldPoints)
% Compute homographies for all images

w1 = warning('Error', 'MATLAB:nearlySingularMatrix'); %#ok
w2 = warning('Error', 'images:maketform:conditionNumberofAIsHigh'); %#ok

numImages = size(points, 3);
validIdx = true(numImages, 1);
homographies = zeros(3, 3, numImages);
for i = 1:numImages
    try    
        homographies(:, :, i) = ...
            computeHomography(double(points(:, :, i)), double(worldPoints));
    catch 
        validIdx(i) = false;
    end
end
warning(w1);
warning(w2);
homographies = homographies(:, :, validIdx);
if ~all(validIdx)
    warning(message('vision:calibrate:invalidHomographies', ...
        numImages - size(homographies, 3), numImages));
end
end

%--------------------------------------------------------------------------
function H = computeHomography(imagePoints, worldPoints)
% Compute projective transformation from worldPoints to imagePoints

H = fitgeotrans(worldPoints, imagePoints, 'projective');
H = (H.T)';
H = H / H(3,3);
end

%----------------------------------------------------------------------------------------------------
function V = computeV(homographies)
% Vb = 0
numImages = size(homographies, 3);
V = zeros(2 * numImages, 6);
for i = 1:numImages
    H = homographies(:, :, i)';
    V(i*2-1,:) = computeLittleV(H, 1, 2);
    V(i*2, :) = computeLittleV(H, 1, 1) - computeLittleV(H, 2, 2);
end
end

%-------------------------------------------------------------------------- 
function v = computeLittleV(H, i, j)
    v = [H(i,1)*H(j,1), H(i,1)*H(j,2)+H(i,2)*H(j,1), H(i,2)*H(j,2),...
         H(i,3)*H(j,1)+H(i,1)*H(j,3), H(i,3)*H(j,2)+H(i,2)*H(j,3), H(i,3)*H(j,3)];
end

%--------------------------------------------------------------------------     
function B = computeB(V)
% lambda * B = inv(A)' * inv(A), where A is the intrinsic matrix

[~, ~, U] = svd(V);
b = U(:, end);

% b = [B11, B12, B22, B13, B23, B33]
B = [b(1), b(2), b(4); b(2), b(3), b(5); b(4), b(5), b(6)];
end


%--------------------------------------------------------------------------
function A = computeIntrinsics(B)
% Compute the intrinsic matrix

cy = (B(1,2)*B(1,3) - B(1,1)*B(2,3)) / (B(1,1)*B(2,2)-B(1,2)^2);
lambda = B(3,3) - (B(1,3)^2 + cy * (B(1,2)*B(1,3) - B(1,1)*B(2,3))) / B(1,1);
fx = sqrt(abs(lambda / B(1,1)));
fy = sqrt(abs(lambda * B(1,1) / (B(1,1) * B(2,2) - B(1,2)^2)));
skew = -B(1,2) * fx^2 * fy / lambda;
cx = skew * cy / fy - B(1,3) * fx^2 / lambda;
A = vision.internal.calibration.constructIntrinsicMatrix(fx, fy, cx, cy, skew);
if ~isreal(A)
    error(message('vision:calibrate:complexCameraMatrix'));
end
end


%--------------------------------------------------------------------------
function [rotationVectors, translationVectors] = ...
    computeExtrinsics(A, homographies)
% Compute translation and rotation vectors for all images

numImages = size(homographies, 3);
rotationVectors = zeros(3, numImages);
translationVectors = zeros(3, numImages); 
Ainv = inv(A);
for i = 1:numImages
    H = homographies(:, :, i);
    h1 = H(:, 1);
    h2 = H(:, 2);
    h3 = H(:, 3);
    lambda = 1 / norm(Ainv * h1); %#ok
    
    % 3D rotation matrix
    r1 = lambda * Ainv * h1; %#ok
    r2 = lambda * Ainv * h2; %#ok
    r3 = cross(r1, r2);
    R = [r1,r2,r3];
    
    rotationVectors(:, i) = vision.internal.calibration.rodriguesMatrixToVector(R);
    
    % translation vector
    t = lambda * Ainv * h3;  %#ok
    translationVectors(:, i) = t;
end
end


%--------------------------------------------------------------------------
function filename = getFileNameFromDir(filedir, patten)
% get file full name from filedir and patten(such as '*.jpg')
imageStruct = dir([filedir  patten]);
numImage = size(imageStruct,1)

filename = cell(numImage,1)

for i=1:numImage
    filename(i)= {strcat(filedir, imageStruct(i).name)}
end
end





