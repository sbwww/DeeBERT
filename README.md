# DeeBERT

This is the code base for the paper [DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference](https://www.aclweb.org/anthology/2020.acl-main.204/).

Code in this repository is also available in the Huggingface Transformer [repo](https://github.com/huggingface/transformers/tree/master/examples/research_projects/deebert) (with minor modification for version compatibility). Check [this page](https://huggingface.co/ji-xin) for models that we have trained in advance (the latest version of Huggingface Transformers Library is needed).

## Installation

Modified and tested on Ubuntu 18.04, Python 3.9, PyTorch 1.10, and Cuda 10.2, following the latest PyTorch installation guide

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# pip3 install torch torchvision torchaudio
```

After installing the required environment, clone this repo, and install the following requirements:

```bash
git clone https://github.com/sbwww/DeeBERT.git
# git clone https://gitee.com/nj-bwshen/DeeBERT.git
cd DeeBERT
conda install --file ./requirements.txt
conda install --file ./examples/requirements.txt
# pip install -r ./requirements.txt
# pip install -r ./examples/requirements.txt
```

---

~~This repo is tested on Python 3.7.5, PyTorch 1.3.1, and Cuda 10.1. Using a virtulaenv or conda environemnt is recommended, for example:~~

~~conda install pytorch==1.3.1 torchvision cudatoolkit=10.1 -c pytorch~~

~~After installing the required environment, clone this repo, and install the following requirements:~~

~~git clone https://github.com/castorini/deebert~~

~~cd deebert~~

~~pip install -r ./requirements.txt~~

~~pip install -r ./examples/requirements.txt~~



## Usage

Added script `DeeBERT_*.sh` to run the entire process

`DeeBERT.ipynb` is also available, but VSCode stucks when the output of tqdm is long, so the notebook is mainly use as a reference of Colab

---

There are four scripts in the `scripts/*/` folder, which can be run from the repo root, e.g., `scripts/glue/train.sh`.

In each script, there are several things to modify before running:

* path to the GLUE dataset. Check [this](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) for more details.
  use the following code to download
  ```bash
  python download_glue_data.py --data_dir glue_data --tasks all
  ```
* path for saving fine-tuned models. Default: `./saved_models`.
* path for saving evaluation results. Default: `./plotting`. Results are printed to stdout and also saved to `npy` files in this directory to facilitate plotting figures and further analyses.
* model_type (bert or roberta)
* model_size (base or large)
* dataset (SST-2, MRPC, RTE, QNLI, QQP, or MNLI)
* settings related to multi-gpu training, `CUDA_VISIBLE_DEVICES` in `scripts/*/*.sh` and `N_GPU` in `DeeBERT_*.sh`.

#### train.sh

This is for fine-tuning and evaluating models as in the original BERT paper.

#### train_highway.sh

This is for fine-tuning DeeBERT models.

The fine-tuning is a two-stage process.

#### eval_highway.sh

This is for evaluating each exit layer for fine-tuned DeeBERT models.

#### eval_entropy.sh

This is for evaluating fine-tuned DeeBERT models, given a number of different early exit entropy thresholds.

## Citation

Please cite the original paper if DeeBERT is used:

```
@inproceedings{xin-etal-2020-deebert,
    title = "{D}ee{BERT}: Dynamic Early Exiting for Accelerating {BERT} Inference",
    author = "Xin, Ji  and
      Tang, Raphael  and
      Lee, Jaejun  and
      Yu, Yaoliang  and
      Lin, Jimmy",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.204",
    pages = "2246--2251",
}
```

Kindly cite this repo if you find it useful:

```
@misc{Shen2021,
  author = {Shen, Bowen},
  title = {sbwww/DeeBERT},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sbwww/DeeBERT}}
}
```
