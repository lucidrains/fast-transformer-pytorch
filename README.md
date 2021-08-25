<img src="./fast-transformer.png" width="400px"></img>

## Fast Transformer - Pytorch

Implementation of <a href="https://arxiv.org/abs/2108.09084">Fast Transformer</a> in Pytorch. This only work as an encoder.

## Install

```bash
$ pip install fast-transformer-pytorch
```

## Usage

```python
import torch
from fast_transformer_pytorch import FastTransformer

model = FastTransformer(
    num_tokens = 20000,
    dim = 512,
    depth = 2,
    max_seq_len = 4096
)

x = torch.randint(0, 20000, (1, 4096))
mask = torch.ones(1, 4096).bool()

logits = model(x, mask = mask) # (1, 4096, 20000)
```

## Citations

```bibtex
@misc{wu2021fastformer,
    title   = {Fastformer: Additive Attention is All You Need}, 
    author  = {Chuhan Wu and Fangzhao Wu and Tao Qi and Yongfeng Huang},
    year    = {2021},
    eprint  = {2108.09084},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```
