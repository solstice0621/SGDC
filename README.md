# SGDC + SGE Modules 

This package provides **two standalone modules** extracted from the original model:

- **SGE**: Structural Guidance Extractor  
  Extracts high-frequency structural priors using Sobel operators and lightweight refinement.

- **SGDC**: Structure-Guided Dynamic Convolution  
  A pooling-free dynamic convolution guided by SGE features.

These components can be plugged into any CNN or Transformer backbone for structure-aware feature modulation.

---

## Usage

### Import
```python
from sge import SGE
from sgdc import SGDC

## Example
sge = SGE(s4_in=2048, s1_in=256)
sgdc = SGDC(dim=256, guide_ch=16)

edge_map, guidance = sge(x4, x1)

output = sgdc(x1, guidance)
