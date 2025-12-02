# SGDC + SGE Modules 

This package provides **two standalone modules** extracted from the original model:

- **SGE**: Structural Guidance Extractor  
  Extracts high-frequency structural priors using Sobel operators and lightweight refinement.

- **SGDC**: Structure-Guided Dynamic Convolution  
  A pooling-free dynamic convolution guided by SGE features.

These components can be plugged into any CNN or Transformer backbone for structure-aware feature modulation.
