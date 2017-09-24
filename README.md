DeepLab
---

This repo contains code to evaluate the models described in the paper:

```
DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs 
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille (*equal contribution) 
Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
```

The code is based on [this](https://github.com/xmyqsh/deeplab-v2) caffe implementation.

The pre-trained models released with the caffe code which have been imported into matconvnet and 
can be downloaded [here](http://www.robots.ox.ac.uk/~albanie/models.html#deeplab-models).

### Functionality

There is a script to evaluate trained models on the `pascal voc 2012` dataset for semantic segmentation.  The training code is still in the verification process.

### Installation

This module can be installed with the following `vl_contrib` commands:

```
vl_contrib('install', 'mcnDeepLab') ;
vl_contrib('setup', 'mcnDeepLab') ;
```  

The code has the following dependencies (these can similarly be added with `vl_contrib`):

* [autonn](https://github.com/vlfeat/autonn) - a wrapper module for matconvnet
* [mcnExtraLayers](https://github.com/albanie/mcnExtraLayers) - extra MatConvNet layers
