% -------------------------------------------------------------
% rectify stereo images all by matlab.
% -------------------------------------------------------------


% add parameters
dir = 'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\';

leftdir = 'binocular-haikang\left\';
rightdir = 'binocular-haikang\right\';
load ([dir, 'binocular-haikang.mat']);
imagetype = '*.png';


% images to process
leftFileDir =  [dir,  leftdir];
rightFileDir = [dir,  rightdir];
% leftFileDir =  [dir,  'binocular-haikang-0\left\']
% rightFileDir = [dir,  'binocular-haikang-0\right\']

imageFileNames1 = getFileNameFromDir(leftFileDir, imagetype);
imageFileNames2 = getFileNameFromDir(rightFileDir, imagetype);

I1 = imread(imageFileNames1{4});
I2 = imread(imageFileNames2{4});
[J1, J2] = rectifyStereoImages(I1, I2, stereoParams,'cubic');

close all;
figure;imshow(J1);
% figure;imshow(J2);

% show disparity map to get max pixel disparity
stereoAnaglyph1 = stereoAnaglyph(J1,J2); imtool(stereoAnaglyph1);


% old methods 16*N disparityRange, 
disparityRange = [0,16 * 12];
tic; disparityMap = disparity(rgb2gray(J1),rgb2gray(J2),'BlockSize',...
    7,'DisparityRange',disparityRange); toc
depth = K_new(1,1) * norm(stereoParams.TranslationOfCamera2) ./ disparityMap;
figure;
imshow(depth,[0,255]);
title('Disparity Map old');
colormap jet
colorbar

% show point clouds
figure;
[xyzPoints, Color] = getPointCloud(J1, disparityMap, stereoParams);
ptCloud = pointCloud(xyzPoints, 'color', Color);
pcshow(ptCloud);

% Use depth to segment objects
mask = repmat((depth > 320 & depth < 390) | ...
              (depth > 440 & depth < 460) | ...
              (depth > 135 & depth < 160),[1,1,3]);
J1(~mask) = 0;
figure;imshow(J1)

% new methods 8*N disparityRange,must be divisible by 8 and must be less than or equal to 128
% tic;disparityMap = disparitySGM(rgb2gray(J1),rgb2gray(J2),'DisparityRange',[0,128], ...
%                             'UniquenessThreshold',20);toc
% depth = K_new(1,1) * norm(T) ./ disparityMap;        
% figure 
% imshow(depth,[0,255]);
% title('Disparity Map new');
% colormap jet
% colorbar


% get file full name from file dir patten(such as '/E:/*.jpg')
function filename = getFileNameFromDir(filedir, patten)
    imageStruct = dir([filedir patten]);
    numImage = size(imageStruct,1);
    
    filename = cell(numImage,1);
    
    for i=1:numImage
        filename(i)= {strcat(filedir, imageStruct(i).name)};
    end
end

% get XYZ-Points using from deep
function [xyz_points, Color] = getPointCloud(I, disparityMap, stereoParams)
    % It's not a general function
    [rows, cols] = size(disparityMap)

    % [u, v, 1]'
    [X, Y] = meshgrid(1:cols,1:rows);
    uv = [X(:), Y(:), ones(rows*cols,1)]';

    % XYZ-Points
    Kl = stereoParams.CameraParameters1.IntrinsicMatrix';
    Kr = stereoParams.CameraParameters2.IntrinsicMatrix';
    K_new = newIntrinsics(Kl, Kr);
    baseLine = stereoParams.TranslationOfCamera2;

    disparityMap = abs(disparityMap);
    depthMap = norm(baseLine) * K_new(1,1) ./ disparityMap;
    new_depthMap = [depthMap(:), depthMap(:), depthMap(:)]';
    xyz_points = (new_depthMap .* (inv(K_new) * uv))';

    % Get rid of infinity, this value is set by myself
    mask = xyz_points(:,1) < 460 & xyz_points(:,1) >= -460 & ...
           xyz_points(:,2) < 460 & xyz_points(:,2) >= -460 & ...
           xyz_points(:,3) < 460 & xyz_points(:,3) >= 0;
    xyz_points(~mask,:) = [];
    
    % add color
    [x,y,z] = size(I);
    Color = reshape(I, x*y, z);
    Color(~mask,:) = [];
end


% get new intrinsic
function K_new = newIntrinsics(Kl, Kr)  
    K_new=Kl;
    
    % find new focal length
    f_new = min([Kr(1,1),Kl(1,1)]);
    
    % set new focal lengths
    K_new(1,1)=f_new; K_new(2,2)=f_new;
    
    % find new y center
    cy_new = (Kr(2,3)+Kl(2,3)) / 2;
    
    % set new y center
    K_new(2,3)= cy_new;
    
    % set the skew to 0
    K_new(1,2) = 0;
end