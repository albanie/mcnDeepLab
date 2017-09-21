classdef Interp < dagnn.Layer
  properties
    zoomFactor = 1
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nninterp(inputs{1}, obj.zoomFactor) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nninterp(inputs{1}, obj.zoomFactor, derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = Interp(varargin)
      obj.load(varargin{:}) ;
      obj.zoomFactor = obj.zoomFactor ;
    end
  end
end
