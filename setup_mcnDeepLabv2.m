function setup_mcnDeepLabv2()
%SETUP_MCNDEEPLABV2 Sets up mcnDeepLabv2, by adding its folders 
% to the Matlab path
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/pascal'], [root '/core']) ;
  addpath([root '/pascal/helpers'], [root '/coco'], [root '/misc']) ;
