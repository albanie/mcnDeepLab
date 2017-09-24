function deeplab_pascal_evaluation(varargin)
%DEEPLAB_PASCAL_EVALUATION evaluate FCN on pascal VOC 2012
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  opts.net = [] ;
  opts.gpus = 1 ;
  opts.modelName = 'deeplab-vggvd-v2' ;
  opts.dataDir = fullfile(vl_rootnn, 'data') ;

  % configure model options
  opts.modelOpts.get_eval_batch = @fcn_eval_get_batch ;
  opts = vl_argparse(opts, varargin) ;

  % load network
  if isempty(opts.net)
    net = deeplab_zoo(opts.modelName) ; 
  else
    net = opts.net ; 
  end

  % evaluation options
  opts.testset = 'val' ; 
  opts.prefetch = true ;

  % configure batch opts
  batchOpts.batchSize = 1 ;
  batchOpts.numThreads = 1 ;
  batchOpts.use_vl_imreadjpeg = true ; 
  batchOpts.imageNeedsToBeMultiple = true ;

  % cache configuration 
  cacheOpts.refreshPredictionCache = false ;
  cacheOpts.refreshDecodedPredCache = false ;
  cacheOpts.refreshEvaluationCache = false ;
  cacheOpts.refreshFigures = false ;

  % configure dataset options
  dataOpts.name = 'pascal' ;
  dataOpts.decoder = 'serial' ;
  dataOpts.getImdb = @(x) getPascalYearImdb(12, x) ; % 2012 data
  dataOpts.displayResults = @displayPascalResults ;
  dataOpts.root = fullfile(vl_rootnn, 'data', 'datasets') ;
  dataOpts.imdbPath = fullfile(opts.dataDir, 'pascal12/standard_imdb/imdb.mat') ;
  dataOpts.configureImdbOpts = @configureImdbOpts ;
  dataOpts.vocEdition = '12' ;
  dataOpts.includeTest = true ;
  dataOpts.includeSegmentation = true ;
  dataOpts.includeDetection = true ;
  dataOpts.vocAdditionalSegmentations = true ;
  dataOpts.dataRoot =  dataOpts.root ;

  % configure paths
  tail = fullfile('evaluations', dataOpts.name, opts.modelName) ;
  expDir = fullfile(vl_rootnn, 'data', tail) ;
  resultsFile = sprintf('%s-%s-results.mat', opts.modelName, opts.testset) ;
  rawPredsFile = sprintf('%s-%s-raw-preds.mat', opts.modelName, opts.testset) ;
  decodedPredsFile = sprintf('%s-%s-decoded.mat', opts.modelName, opts.testset) ;
  evalCacheDir = fullfile(expDir, 'eval_cache') ;

  cacheOpts.rawPredsCache = fullfile(evalCacheDir, rawPredsFile) ;
  cacheOpts.decodedPredsCache = fullfile(evalCacheDir, decodedPredsFile) ;
  cacheOpts.resultsCache = fullfile(evalCacheDir, resultsFile) ;
  cacheOpts.evalCacheDir = evalCacheDir ;

  % configure meta options
  opts.dataOpts = dataOpts ;
  opts.batchOpts = batchOpts ;
  opts.cacheOpts = cacheOpts ;
  deeplab_evaluation(expDir, net, opts) ;

% -----------------------------------------------------------
function opts = configureImdbOpts(expDir, opts)
% -----------------------------------------------------------
% configure VOC options 
% (must be done after the imdb is in place since evaluation
% paths are set relative to data locations)

opts.dataOpts = configureVOC(expDir, opts.dataOpts, 'test') ;

%-----------------------------------------------------------
function dataOpts = configureVOC(expDir, dataOpts, testset) 
%-----------------------------------------------------------
% LOADPASCALOPTS Load the pascal VOC database options
%
% NOTE: The Pascal VOC dataset has a number of directories 
% and attributes. The paths to these directories are 
% set using the VOCdevkit code. The VOCdevkit initialization 
% code assumes it is being run from the devkit root folder, 
% so we make a note of our current directory, change to the 
% devkit root, initialize the pascal options and then change
% back into our original directory 

VOCRoot = fullfile(dataOpts.dataRoot, 'VOCdevkit2007') ;
VOCopts.devkitCode = fullfile(VOCRoot, 'VOCcode') ;

% check the existence of the required folders
assert(logical(exist(VOCRoot, 'dir')), 'VOC root directory not found') ;
assert(logical(exist(VOCopts.devkitCode, 'dir')), 'devkit code not found') ;

currentDir = pwd ; cd(VOCRoot) ; addpath(VOCopts.devkitCode) ;
VOCinit ; % VOCinit loads database options into a variable called VOCopts

dataDir = fullfile(VOCRoot, '2007') ;
VOCopts.localdir = fullfile(dataDir, 'local') ;
VOCopts.imgsetpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
VOCopts.imgpath = fullfile(dataDir, 'ImageSets/Main/%s.txt') ;
VOCopts.annopath = fullfile(dataDir, 'Annotations/%s.xml') ;
VOCopts.cacheDir = fullfile(expDir, '2007/Results/Cache') ;
VOCopts.drawAPCurve = false ;
VOCopts.testset = testset ;
detDir = fullfile(expDir, 'VOCdetections') ;

% create detection and cache directories if required
requiredDirs = {VOCopts.localdir, VOCopts.cacheDir, detDir} ;
for i = 1:numel(requiredDirs)
    reqDir = requiredDirs{i} ;
    if ~exist(reqDir, 'dir') , mkdir(reqDir) ; end
end

VOCopts.detrespath = fullfile(detDir, sprintf('%%s_det_%s_%%s.txt', 'test')) ;
dataOpts.VOCopts = VOCopts ;
cd(currentDir) ; % return to original directory

% ---------------------------------------------------------------------------
function displayPascalResults(modelName, aps, opts)
% ---------------------------------------------------------------------------

fprintf('\n============\n') ;
fprintf(sprintf('%s set performance of %s:', opts.testset, modelName)) ;
fprintf('%.1f (mean ap) \n', 100 * mean(aps)) ;
fprintf('\n============\n') ;
