function y = vl_nninterp(x, zoom, varargin)
% VL_NNINTERP Dynamic bilinear interpolation
%   Y = VL_NNINTERP(X, ZOOM) a simple wrapper to apply bilinear interpolation
%   to achieve a given ZOOM factor for an input tensor X.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  inSz = [size(x, 1) size(x, 2)] ;
  outSz = ((inSz-1) * double(zoom)) + 1 ;

  % generate sampling grid (should probably cache this)
  useGPU = isa(x, 'gpuArray') ;
  Ho = outSz(1) ; Wo = outSz(2) ;
  xi = linspace(-1, 1, Ho) ;
  yi = linspace(-1, 1, Wo) ;
  [yy, xx] = meshgrid(single(xi), single(yi)) ;
  xxyy = [yy(:), xx(:)] ; % Mx2
  if useGPU, xxyy = gpuArray(xxyy) ; end
  grid = reshape(xxyy, Wo, Ho, 2) ;
  grid = permute(grid, [3,2,1]) ;

 if isempty(dzdy)
    y = vl_nnbilinearsampler(x, grid) ;
 else
   y = vl_nnbilinearsampler(x, grid, dzdy{1}) ;
 end
