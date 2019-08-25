% -------------------------------------------------------------
% rectify stereo images all by myself.
% -------------------------------------------------------------


dir = 'E:\Course-Learning\computer_vision\camera_calibration\calibration_matlab\data\';

leftdir = 'binocular-haikang\left\';
rightdir = 'binocular-haikang\right\';
load ([dir, 'binocular-haikang.mat']);
imagetype = '*.png';


% images to process
leftFileDir =  [dir,  leftdir];
rightFileDir = [dir,  rightdir];

imageFileNames1 = getFileNameFromDir(leftFileDir, imagetype);
imageFileNames2 = getFileNameFromDir(rightFileDir, imagetype);

% undistort image
image_left  =  imread(imageFileNames1{2});
image_right  = imread(imageFileNames2{2});

undistort_image_left = undistortImage(image_left, stereoParams.CameraParameters1, 'OutputView','same');
undistort_image_right = undistortImage(image_right, stereoParams.CameraParameters2, 'OutputView','same');

% figure; imshowpair(image_left,undistort_image_left,'montage');
% figure; imshowpair(image_right,undistort_image_right,'montage');

% stereo rectify matrix
R = stereoParams.RotationOfCamera2';
T = stereoParams.TranslationOfCamera2'; % Rotate first, then translate

% rotate half
r  = vision.internal.calibration.rodriguesMatrixToVector(R);
R1 = vision.internal.calibration.rodriguesVectorToMatrix(r/2);
R2 = vision.internal.calibration.rodriguesVectorToMatrix(-r/2);

% rotate -r/2 relative to camera2 
T = R2 * T;

% alignment
Rrect = alignmentRotation(T);
Rrect1 = Rrect * R1;
Rrect2 = Rrect * R2;

% intrinsic
Kl = stereoParams.CameraParameters1.IntrinsicMatrix';
Kr = stereoParams.CameraParameters2.IntrinsicMatrix';

% new camera intrinsic
K_new = newIntrinsics(Kl, Kr);  

Map1_real2new = Kl * Rrect1' * inv(K_new);
Map2_real2new = Kr * Rrect2' * inv(K_new);

% ramap
[row, col, ~] = size(undistort_image_left);

new_points = ones(3, 1);
new_image_1 = zeros(size(undistort_image_left), 'uint8');
new_image_2 = zeros(size(undistort_image_right), 'uint8');
 
% search origin pixel
% left picture
for i = 1:row
    for j = 1:col
        new_points(2) = i;
        new_points(1) = j;

        real_points = Map1_real2new * new_points;
        real_points = real_points / real_points(3);
        
        x = real_points(1);
        y = real_points(2);

        if (x >= 1) & (x <= col) & ...
           (y >= 1) & (y <= row) 

            new_image_1(i,j,:) = undistort_image_left(round(y), round(x),:);
        end
    end
end

% right picture
for i = 1:row
    for j = 1:col
        new_points(2) = i;
        new_points(1) = j;

        real_points = Map2_real2new * new_points;
        real_points = real_points / real_points(3);
        
        x = real_points(1);
        y = real_points(2);

        if (x >= 1) & (x <= col) & ...
           (y >= 1) & (y <= row) 
           new_image_2(i,j,:) = undistort_image_right(round(y), round(x),:);
        end
    end
end

figure; imshow(new_image_1);
figure; imshow(new_image_2);

% old methods 16*N disparityRange
disparityRange = [0,16 * 14];
disparityMap = disparity(rgb2gray(new_image_1),rgb2gray(new_image_2),'BlockSize',...
    7,'DisparityRange',disparityRange);

depth = K_new(1,1) * norm(T) ./ disparityMap;

figure;
imshow(depth,[0,255]);
title('Disparity Map old');
colormap jet
colorbar



% % get rectify Matrix form T of camera1 to camera2 two
% function Rrect = rectifyMatrix(T)
%     % acos(T, [1,0,0]) must greater than zero
%     e1 = T ./ norm(T); 
%     e2 = [-e1(2), e1(1), 0]' ./ norm([-e1(2), e1(1), 0]');
%     e3 = cross(e1, e2) ./ norm(cross(e1, e2));
%     Rrect = [e1,e2,e3]';
% end

%----------------------------------------------------------------
% get file full name from file dir patten(such as '/E:/*.jpg')
function filename = getFileNameFromDir(filedir, patten)
imageStruct = dir([filedir patten]);
numImage = size(imageStruct,1);

filename = cell(numImage,1);

for i=1:numImage
    filename(i)= {strcat(filedir, imageStruct(i).name)};
end
end

%----------------------------------------------------------------
% draw line to image
function line_image = drawLine(image, interval)
    [row, col, ~] = size(image);

    line_image = image;
    for i = 1:interval:row
        for j = 1:col
            line_image(i, j, :) = [255,0,0];
        end
    end
end

%----------------------------------------------------------------
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


%----------------------------------------------------------------
function RrowAlign = alignmentRotation(t)
    % RrowAlign represents some point rotated from xUnitVector to t, which 
    % is equivalent to the coordinate axis rotated from t to xUnitVector.
    % t must be alone Ol->Or derection.
    xUnitVector = [1;0;0];
    if dot(xUnitVector, t) < 0
    % xUnitVector = -xUnitVector;
          t = -t;
    end
    
    % find the axis of rotation
    rotationAxis = cross(t,xUnitVector);
    
    if norm(rotationAxis) == 0 % no rotation
        RrowAlign = eye(3);
    else
        rotationAxis = rotationAxis / norm(rotationAxis);
        
        % find the angle of rotation
        angle = acos(abs(dot(t,xUnitVector))/(norm(t)*norm(xUnitVector)));
        
        rotationAxis = angle * rotationAxis;
        
        % convert the rotation vector into a rotation matrix
        RrowAlign = vision.internal.calibration.rodriguesVectorToMatrix(rotationAxis);
    end
end