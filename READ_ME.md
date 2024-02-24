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
@misc{van2024policy,
      title={Policy Learning for Off-Dynamics RL with Deficient Support}, 
      author={Linh Le Pham Van and Hung The Tran and Sunil Gupta},
      year={2024},
      eprint={2402.10765},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
