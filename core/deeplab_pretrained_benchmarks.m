function deeplab_pretrained_benchmarks(varargin)
% DEEPLAB_PRETRAINED_BENCHMARKS evalute public models
%   DEEPLAB_PRETRAINED_BENCHMARKS evalute the publicly released
%   deeplab models on the Pascal 2012 semantic segmentation validation
%   set.
%
%   DEEPLAB_PRETRAINED_BENCHMARKS(..., 'option', value, ...) accepts 
%   the following options:
%
%   `gpus`:: []
%    Device on which to run network 
%
%   NOTE: Some of the models listed below used the validation set for 
%   training - in these cases the scores can be useful to assess overfitting.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.gpus = 1 ;
  opts = vl_argparse(opts, varargin) ;

  models = {...
      'deeplab-vggvd-t-v2' ...
      'deeplab-vggvd-v2' ...
      'deeplab-res101-t-v2' ...
      'deeplab-res101-v2' ...
      } ;

  for ii = 1:numel(models)
      model = models{ii} ;
      deeplab_pascal_evaluation('modelName', model, 'gpus', opts.gpus) ;
  end

