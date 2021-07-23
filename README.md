# AdvOrder: Practical Relative Order Attack

**Paper Title:** Practical Relative Order Attack in Deep Ranking, ICCV'2021
**Preprint Link:** https://arxiv.org/abs/2103.05248

## General Usage

| Step # | Description                          | Command                                  |
| ---    | ---                                  | ---                                      |
| 1      | Download the datasets                | `python3 Download.py`                    |
| 2.1    | Train network on fasion              | `python3 Train.py -D cuda -M faC_c2f2`   |
| 2.2    | Train network on sop                 | `python3 Train.py -D cuda -M sopE_res18` |
| 3      | Conduct white-box attack experiments | `bash bin/wtable`                        |
| 4.1.1  | (black box) Rand on fasion           | `bash bin/farandsearch 5 50 4`           |
| 4.1.2  | (black box) Beta on fasion           | `bash bin/fabatk5 50 4`                  |
| 4.1.3  | (black box) PSO  on fasion           | `bash bin/fapso 5 50 4`                  |
| 4.1.4  | (black box) NES  on fasion           | `bash bin/fanes 5 50 4`                  |
| 4.1.5  | (black box) SPSA on fasion           | `bash bin/faspsa 5 50 4`                 |
| 4.2.1  | (black box) SPSA on SOP              | `bash bin/randsearch 5 50 4`             |
| 4.2.2  | (black box) SPSA on SOP              | `bash bin/batk 5 50 4`                   |
| 4.2.3  | (black box) SPSA on SOP              | `bash bin/pso 5 50 4`                    |
| 4.2.4  | (black box) SPSA on SOP              | `bash bin/nes 5 50 4`                    |
| 4.2.5  | (black box) SPSA on SOP              | `bash bin/spsa 5 50 4`                   |

The syntax of the `bin/*` black-box attack commands is `bin/<algorithm> k N varepsilon*255`.

Hint: `export USE_CPP_KERNEL=1` can significantly speed up the black-box attack experiments.

Hint: `export USE_RUST_KERNEL=1` will use the Rust implementation of SRC which is even faster than the C++ one. Requires `rustc`.

Hint: Enabling Adam optimizer for SPSA (`export SS_ADAM`) may slightly boost the performance.

## Detailed File Descriptions

```
.
├── Attack.py             | entrance script for white-box order attack
├── bin                   | collection of shortcut scripts
│   ├── batk              |   Beta on sop
│   ├── batkparam         |   Beta parameter search example
│   ├── batktable         |   Beta batched experiments
│   ├── fabatk            |   Beta on fashion
│   ├── fanes             |   NES  on fashion
│   ├── fapso             |   PSO  on fashion
│   ├── fapsops           |   PSO  parameter search example
│   ├── farandsearch      |   Rand on fashion
│   ├── faspsa            |   SPSA on fashion
│   ├── nes               |   NES  on sop
│   ├── nesparam          |   NES  parameter search example
│   ├── nestable          |   NES  batched experiments
│   ├── nodrtable         |   Ablation: no dimension reduction
│   ├── pso               |   PSO  on sop
│   ├── psoparam          |   PSO  parameter search example
│   ├── psotable          |   PSO  batched experiments
│   ├── qbudcurve.py      |   plot a figure in appendix
│   ├── randsearch        |   Rand on sop
│   ├── randsearchtable   |   Rand batched experiments
│   ├── spsa              |   SPSA on sop
│   ├── spsaparam         |   SPSA parameter search example
│   ├── spsatable         |   SPSA batched experiments
│   ├── wloss.py          |   plot a figure in main manuscript
│   ├── wloss-sop.py      |   plot a figure in appendix
│   └── wtable            |   white-box batched experiments
├── BlackOA.py            | entrance script for black-box order attack
├── config.yml            | configuration file for models and attacks
├── display.py            | display query results from SnapShop
├── _download.py          | helper utility used for download files
├── Download.py           | download the fashion minst dataset
├── lib                   | core algorithms
│   ├── common.py         |   white-box order attack implementation
│   ├── datasets          |   dataset abstractions
│   │   ├── fashion.py    |     fashion-mnist dataset
│   │   ├── __init__.py   |     python file
│   │   └── sop.py        |     stanford-online-products
│   ├── faC_c2f2.py       |   fashion c2f2 network with cosine metric
│   ├── faC_lenet.py      |   fashion lenet with cosine metric
│   ├── faC_res18.py      |   fashion resnet18 
│   ├── faE_c2f2.py       |   fashion c2f2 network with euclidean metric
│   ├── __init__.py       |   python file
│   ├── rankingmodel.py   |   abstract class
│   ├── reorder.py        |   black-box order attack implementation
│   ├── snapshop.py       |   snapshop client and abstraction
│   ├── sopE_res18.py     |   sop resnet18 with euclidean metric
│   ├── sopE_res50.py     |   sop resnet50 with euclidean metric
│   ├── srckernel_py.py   |   Primitive SRC implementation in python. (slow) 
│   ├── _srckernel.cc     |   SRC function in C++ (moderate speed)
│   ├── srckernel_cc.py   |   python wrapper for the C++ SRC function
│   ├── srck/*            |   Rust implementation of the SRC function. (fast)
│   ├── srckernel_rs.py   |   python wrapper for the Rust SRC function
│   ├── test_srckernel.py |   tester of the C++ SRC function
│   └── utils.py          |   miscellaneous
├── poc                   |   miscellaneous
│   └── taumap.py         |   miscellaneous
├── PracticalOA.py        | entrance script for practical order attack
├── Train.py              | entrance script for training ranking model
└── visrow.py             | visualization helper for snapshop attack
```

### Software Version

```
Python  3.8.3
PyTorch 1.7.0
Numpy   1.18.5
Scipy   1.5.0
CUDA    11
Cargo   1.45.0
Rustc   1.48.0
Linux   5.10
```

### License Info

```
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
```
