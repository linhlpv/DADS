# Policy Learning for Off-Dynamics RL with Deficient Support

This is the source code for replicating the results from our paper accepted to **AAMAS'24**, titled **['Policy Learning for Off-Dynamics RL with Deficient Support'](https://arxiv.org/abs/2402.10765)**.

Thank you for your interest!

## Setup
Before training, please install the following packages and libraries by running the following command
```
conda create --name dads
conda activate dads
pip install -r requirements.txt
```

## Training 
Implementation for envs is in the `defficient_support_mujoco_noise_envs.py` and DADS in `dads.py`
Run the command below to train the DADS agents
```
python dads.py --device=cuda
```

## Citation

If you find our code helpful or utilise our proposed method as comparison baselines in your experiments, please cite our paper. Again, thank you for your interest!
```
@inproceedings{10.5555/3635637.3662965,
author = {Le Pham Van, Linh and The Tran, Hung and Gupta, Sunil},
title = {Policy Learning for Off-Dynamics RL with Deficient Support},
year = {2024},
isbn = {9798400704864},
publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
pages = {1093–1100},
series = {AAMAS '24}
}
