
x = [1,1,0]';
T = [0, 1, 0]';
r = [0, 0, pi/2]';
R = vision.internal.calibration.rodriguesVectorToMatrix(r);

% relationship of the translation and rotation between coordinates points and coordinates frame
% the coordinate system doesn't move, but the coordinate points moves

x11 = R*x           % x is rotated positive 90 degrees around the z-axis
x12 = R*x + T       % after rotation, translate it by T in the original frame

x21 = x+T           % translate it by T in the original frame
x22 = R*(x + T)     % after translation, rotate it by R in the original frame

% 2.the coordinate points doesn't move, but the coordinate system moves
x31 = R'*x          % coordinate system is rotated positive 90 degrees around the z-axis
x32 = R'*x - T      % after rotation, coordinate system translation -T  in the new coordinate system

x41 = x - T         % coordinate system rotate positive 90 degrees around coordinate system 
X42 = R'*(x-T)      % after translation, rotate it by R in the new coordinate system 


