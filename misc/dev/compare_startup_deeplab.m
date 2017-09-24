% reload paths etc.
% NOTE!
% The mex lock has been removed from caffe_.cpp! 
%munlock('caffe_') ;
clear all ; %#ok
caffePath = '/users/albanie/coding/libs/caffes/deeplabv2-caffe/matlab' ;
%munlock('caffe_') ;
%clear caffe_.mexa64 ;
clear caffe_ ;
rehash path ;

% hack to avoid cuda errors
a = gpuArray(1) ; clear a ; %#ok

% refresh/set up ssd-caffe
addpath(caffePath) ;

caffeRoot = '/users/albanie/coding/libs/caffes/deeplabv2-caffe' ;

% set paths and load caffe ssd-model
modelDir = '/users/albanie/data/models/caffe/deeplab/res101' ;
dataDir = '/users/albanie/data/models/caffe/deeplab/res101' ;
model = fullfile(modelDir, 'safe_test.prototxt') ;
weights = fullfile(dataDir, 'train_iter_20000.caffemodel') ;
caffe.set_mode_cpu() ;

% to use the relative paths defined in the prototxt, we change into the 
% caffe root to load the network
cd(caffeRoot) ;

% load model
caffeNet = caffe.Net(model, weights, 'test') ;

% load mcn model for comparison
opts.trunkModelPath = fullfile(vl_rootnn, 'data/models-import', ...
                                'deeplab-res101-t-v2.mat') ;
opts.imageSize = [513 513] ;
net = load(opts.trunkModelPath) ; net = dagnn.DagNN.loadobj(net) ;
