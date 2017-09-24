function fix_imports_deeplab(varargin)
%FIX_SSD_IMPORTS - clean up imported caffe models
%   FIX_SSD_IMPORTS performs some additional clean up work
%   on models imported from caffe to ensure that they are
%   consistent with matconvnet conventions. 
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  %opts.imdbPath = fullfile(vl_rootnn, 'data/imagenet12/imdb.mat') ;
  opts.numClasses = 21 ;
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts = vl_argparse(opts, varargin) ;

  %imdb = load(opts.imdbPath) ;

  % select model
  res = dir(fullfile(opts.modelDir, '*.mat')) ; modelNames = {res.name} ;
  modelNames = modelNames(contains(modelNames, 'deeplab')) ;

  for mm = 1:numel(modelNames)
    modelPath = fullfile(opts.modelDir, modelNames{mm}) ;
    fprintf('fixing name scheme for %s\n', modelNames{mm}) ;
    net = load(modelPath) ; 

    % fix naming convention
    for ii = 1:numel(net.layers)
      net.layers(ii).name = strrep(net.layers(ii).name, '/', '_') ;
      net.layers(ii).inputs = strrep(net.layers(ii).inputs, '/', '_') ;
      net.layers(ii).outputs = strrep(net.layers(ii).outputs, '/', '_') ;
      net.layers(ii).params = strrep(net.layers(ii).params, '/', '_') ;

      % fix layer option types
      if strcmp(net.layers(ii).type, 'dagnn.Interp')
        block = net.layers(ii).block ;
        net.layers(ii).block.zoomFactor = double(block.zoomFactor) ;
        net.layers(ii).block.shrinkFactor = double(block.shrinkFactor) ;
        net.layers(ii).block.padBeg = double(block.padBeg) ;
        net.layers(ii).block.padEnd = double(block.padEnd) ;
      end
    end
    for ii = 1:numel(net.params)
      net.params(ii).name = strrep(net.params(ii).name, '/', '_') ;
    end

    % fix meta 
    fprintf('adding info to %s (%d/%d)\n', modelPath, mm, numel(modelNames)) ;
    %net.meta.classes = imdb.classes ;
    net.meta.normalization.averageImage = [122.675 116.669 104.008] ;
    net.meta.normalization.imageSize = [513 513 3] ;
    if contains(modelNames{mm}, 'res101')
      net.meta.predVar = 'fc1_interp' ;
    elseif contains(modelNames{mm}, 'vggvd')
      net.meta.predVar = 'fc8_interp' ;
    else, error('%s trunk not recognised', modelNames{mm}) ;
    end
    net = dagnn.DagNN.loadobj(net) ; 
    net = net.saveobj() ; save(modelPath, '-struct', 'net') ; %#ok
  end
