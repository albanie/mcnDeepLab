function imdb = getPascal12Imdb(opts) 
% GETPASCAL12IMDB - load Pascal Imdb file, making use of the vocSetup
% code by Sebastien Erhardt and Andrea Vedaldi
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  dataDir = fullfile(opts.dataOpts.root, 'pascal12') ;
  imdb = vocSetup('dataDir', dataDir, ...
    'edition', opts.dataOpts.vocEdition, ...
    'includeTest', opts.dataOpts.includeTest, ...
    'includeSegmentation', opts.dataOpts.includeSegmentation, ...
    'includeDetection', opts.dataOpts.includeDetection) ;

  if opts.dataOpts.vocAdditionalSegmentations
    imdb = vocSetupAdditionalSegmentations(imdb, 'dataDir', opts.dataDir) ;
  end
