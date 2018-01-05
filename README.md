#### graph2vec

#### Installation

```
conda create -n g2v_env python=2.7 pip -y
source activate vn_env
pip install -r requirements.txt
```

#### Usage

See `./run.sh` for usage.

See [graph2vec paper](https://arxiv.org/pdf/1707.05005.pdf) for information on algorithm.
See [WL Graph Kernels paper](http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf) for information on WL Kernel, which is the core graph operation at play here.

See based on [graph2vec_tf code](https://github.com/MLDroid/graph2vec_tf) for original code base.

