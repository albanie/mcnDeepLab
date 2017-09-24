function info = deeplab_evaluation(expDir, net, opts)

  % Setup data
  if exist(opts.dataOpts.imdbPath, 'file')
    imdb = load(opts.dataOpts.imdbPath) ;
  else
    imdb = opts.dataOpts.getImdb(opts) ;
    imdbDir = fileparts(opts.dataOpts.imdbPath) ;
    if ~exist(imdbDir, 'dir'), mkdir(imdbDir) ; end
    save(opts.dataOpts.imdbPath, '-struct', 'imdb') ;
  end

  switch opts.testset
    case 'val', setLabel = 2 ;
    case 'test', setLabel = 3 ;
  end

  testIdx = find(imdb.images.set == setLabel & imdb.images.segmentation) ;


% Compare the validation set to the one used in the FCN paper
% valNames = sort(imdb.images.name(val)') ;
% valNames = textread('data/seg11valid.txt', '%s') ;
% valNames_ = textread('data/seg12valid-tvg.txt', '%s') ;
% assert(isequal(valNames, valNames_)) ;

% -------------------------------------------------------------------------
% Run evaluation
% -------------------------------------------------------------------------
mkdirRecursive(expDir) ;
prepareGPUs(opts, true) ;

if ~isempty(opts.gpus) && strcmp(net.device, 'cpu'), net.move('gpu') ; end

net.mode = 'test' ;
confusion = zeros(21) ;
predIdx = net.getVarIndex(net.meta.predVar) ;

for ii = 1:numel(testIdx)
  imId = testIdx(ii) ;
  name = imdb.images.name{imId} ;
  rgbPath = sprintf(imdb.paths.image, name) ;
  labelsPath = sprintf(imdb.paths.classSegmentation, name) ;

  % Load an image and gt segmentation
  rgb = vl_imreadjpeg({rgbPath}) ;
  rgb = rgb{1} ;
  anno = imread(labelsPath) ;
  lb = single(anno) ;
  lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg

  % Subtract the mean (color)
  meanIm = net.meta.normalization.averageImage ;
  if numel(meanIm) == 3
    meanIm = reshape(meanIm, [1 1 3]) ;
  end
  im = bsxfun(@minus, single(rgb), meanIm) ;

  % Some networks requires the image to be a multiple of 32 pixels
  if opts.batchOpts.imageNeedsToBeMultiple
    sz = [size(im,1), size(im,2)] ;
    sz_ = round(sz / 32)*32 ;
    im_ = imresize(im, sz_) ;
  else
    im_ = im ;
  end

  if ~isempty(opts.gpus), im_ = gpuArray(im_) ; end

  net.eval({'data', im_}) ;
  scores_ = gather(net.vars(predIdx).value) ;
  [~,pred_] = max(scores_,[],3) ;

  if opts.batchOpts.imageNeedsToBeMultiple
    pred = imresize(pred_, sz, 'method', 'nearest') ;
  else
    pred = pred_ ;
  end

  % Accumulate errors
  ok = lb > 0 ;
  confusion = confusion + accumarray([lb(ok),pred(ok)],1,[21 21]) ;

  % Plots
  if mod(ii - 1,100) == 0 || ii == numel(testIdx)
    clear info ;
    [info.iu, info.miu, info.pacc, info.macc] = getAccuracies(confusion) ;
    fprintf('IU ') ;
    fprintf('%4.1f ', 100 * info.iu) ;
    fprintf('\n(%d/%d) meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
            ii, numel(testIdx), 100*info.miu, 100*info.pacc, 100*info.macc) ;

    figure(1) ; clf;
    imagesc(normalizeConfusion(confusion)) ;
    axis image ; set(gca,'ydir','normal') ;
    colormap(jet) ;
    drawnow ;

    % Print segmentation
    figure(100) ;clf ;
    displayImage(rgb/255, lb, pred) ;
    drawnow ;

    % Save segmentation
    imPath = fullfile(expDir, [name '.png']) ;
    imwrite(pred,labelColors(),imPath,'png');
  end
end

% Save results
resPath = opts.cacheOpts.resultsCache ;
mkdirRecursive(fileparts(resPath)) ;
save(resPath, '-struct', 'info') ;

% ------------------------------
function mkdirRecursive(dirname)
% ------------------------------

  fprintf('at %s\n', dirname) ;
  while ~exist(fileparts(dirname), 'dir')
    fprintf('at %s\n', dirname) ;
    mkdirRecursive(fileparts(dirname)) ;
  end
  if ~exist(dirname, 'dir'), mkdir(dirname) ; end

% -------------------------------------------------------------------------
function nconfusion = normalizeConfusion(confusion)
% -------------------------------------------------------------------------
% normalize confusion by row (each row contains a gt label)
nconfusion = bsxfun(@rdivide, double(confusion), double(sum(confusion,2))) ;

% -------------------------------------------------------------------------
function [IU, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
meanIU = mean(IU) ;
pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
meanAccuracy = mean(tp ./ max(1, pos)) ;

% -------------------------------------------------------------------------
function displayImage(im, lb, pred)
% -------------------------------------------------------------------------
subplot(2,2,1) ;
image(im) ;
axis image ;
title('source image') ;

subplot(2,2,2) ;
image(uint8(lb-1)) ;
axis image ;
title('ground truth')

cmap = labelColors() ;
subplot(2,2,3) ;
image(uint8(pred-1)) ;
axis image ;
title('predicted') ;

colormap(cmap) ;

% -------------------------------------------------------------------------
function cmap = labelColors()
% -------------------------------------------------------------------------
N=21;
cmap = zeros(N,3);
for i=1:N
  id = i-1; r=0;g=0;b=0;
  for j=0:7
    r = bitor(r, bitshift(bitget(id,1),7 - j));
    g = bitor(g, bitshift(bitget(id,2),7 - j));
    b = bitor(b, bitshift(bitget(id,3),7 - j));
    id = bitshift(id,-3);
  end
  cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    gpuDevice(opts.gpus)
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;
