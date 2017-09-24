function deeplab_pretrained_benchmarks(expIdx)
% DEEPLAB_PRETRAINED_BENCHMARKS evalute the publicly released
% deeplab models on the Pascal 2012 semantic segmentation validation
% set.
%
%   NOTE: Some of the models listed below used the validation set for 
%   training - in these cases the scores can be useful to assess overfitting.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

switch expIdx
  case 1, gpu = 2 ; models = {'deeplab-vggvd-t-v2'} ;
  case 2, gpu = 3 ; models = {'deeplab-vggvd-v2'} ;
end

%models = {...
    %'deeplab-vggvd-t-v2' ...
    %'deeplab-vggvd-v2' ...
    %'deeplab-res101-t-v2' ...
    %'deeplab-res101-v2' ...
    %} ;

for ii = 1:numel(models)
    model = models{ii} ;
    deeplab_pascal_evaluation('modelName', model, ...
                              'gpus', gpu) ;
end

