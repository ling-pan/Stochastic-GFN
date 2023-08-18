# Stochastic Generative Flow Networks

This repository is the implementation of [Stochastic Generative Flow Networks](https://openreview.net/fohttps://proceedings.mlr.press/v216/pan23a/pan23a.pdf) in UAI 2023 (Spotlight). This codebase is based on the open-source [gflownet](https://github.com/GFNOrg/gflownet) implementation and [BioSeq-GFN-AL](https://github.com/MJ10/BioSeq-GFN-AL) implementation, and please refer to those repos for more documentation.

## Citing

If you used this code in your research or found it helpful, please consider citing our paper:
```
@inproceedings{
	pan2023stochastic,
	title={Stochastic Generative Flow Networks},
	author={Ling Pan and Dinghuai Zhang and Moksh Jain and Longbo Huang and Yoshua Bengio},
	booktitle={Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence}ce on Learning Representations},
	year={2023},
	url={https://proceedings.mlr.press/v216/pan23a.html}
}
```

## Requirements

### Grid
- python: 3.6
- torch: 1.3.0
- scipy: 1.5.4
- numpy: 1.19.5
- tdqm

### Sequence
Please check the [BioSeq-GFN-AL](https://github.com/MJ10/BioSeq-GFN-AL) repo for more details about the environment.

## Usage

Please follow the instructions below to replicate the results in the paper. 
- Grid (in the grid folder)
```
python main.py --stick <STICK> --horizon <HORIZON> --seed <SEED>
```
- Sequence (in the tfb folder)
```
python run_tfbind.py --stick <STICK> --seed <SEED>
```
