function y = vl_nninterp(x, shrink, zoom, varargin)
% VL_NNINTERP Dynamic bilinear interpolation
%   Y = VL_NNINTERP(X, ZOOM) a simple wrapper to apply bilinear interpolation
%   to achieve a given SHRINK/ZOOM factor for an input tensor X in a manner 
%   that is compatible with the caffe DEEPLAB framework.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.padBeg = 0 ;
  opts.padEnd = 0 ;
  [opts, dzdy] = vl_argparsepos(opts, varargin) ;

  % determine output size
  inSz = [size(x, 1) size(x, 2)] ;
  inSz = inSz + opts.padBeg + opts.padEnd ;
  outSz = round((inSz - 1) / shrink) + 1 ;
  outSz = outSz + (outSz -1) * (zoom - 1) ;
  %outSz = ((inSz-1) * double(zoom)) + 1 ;

  % generate sampling grid (should probably cache this)
  useGPU = isa(x, 'gpuArray') ;
  Ho = outSz(1) ; Wo = outSz(2) ;
  xi = linspace(-1, 1, Ho) ; yi = linspace(-1, 1, Wo) ;
  [yy, xx] = meshgrid(single(xi), single(yi)) ;
  xxyy = [yy(:), xx(:)] ;
  if useGPU, xxyy = gpuArray(xxyy) ; end
  grid = reshape(xxyy, Wo, Ho, 2) ;
  grid = permute(grid, [3,2,1]) ;

 if isempty(dzdy)
    y = vl_nnbilinearsampler(x, grid) ;
 else
   y = vl_nnbilinearsampler(x, grid, dzdy{1}) ;
 end
