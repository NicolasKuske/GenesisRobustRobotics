# Robust Multimodal Continual Learning for Robotics

This repository contains RL environments using the Genesis general-purpose physics platform to test multimodal continual learning.

You can choose between xyz-position based RL and vision-based RL. --- currently multimodality is not implemented


## 🔥 News

- [2025-06-17] Revolving task-model classes
- [2025-06-14] Focus on the bare necessities
- [2025-06-13] Set up the repository

  
## Requirements

Please install Pytorch.


You get the Genesis dependencies via: 
```
pip install genesis-world
```
That's it for now!


## Command-line Arguments

- `-v` or `--vis` enables visualization.
- `-l` or `--load_path` specifies the loading path of a previously saved model checkpoint. Do **not** include this argument if you intend to train your model from scratch.
- `-n` or `--num_envs` specifies the number of parallel environments. If none is provided, the default is `1`.
- `-b` or `--batch_size` defines the batch size used for training. If none is provided, the default is `64 * num_envs`.
- `-t` or `--task` specifies the task to train on. If none is provided, the default is `GraspFixedBlock`. Available tasks include:
  - `GraspFixedBlock`: Environment for grasping a fixed block.
  - `GraspRandomBlock`: Environment for grasping a randomly placed block.



## Usage

- Training

You can run different learning algorithms with the following command structure. Here is an example of running training with 10 envs using xyz-position based RL 
```bash
python run_ppo_position.py -n 10
```
Exchange 'pos' with 'vis' for vision based RL. 


<img  src="figs/train.gif" width="300">

- Evaluation

To test the trained policy, you can load a pretrained model from the checkpoint (if one has been saved) and visualize the rollout, by executing the script with the following command-line arguments:
```bash
python run_ppo_{algo}.py -v -l `logs/{task}_{algo}_checkpoint_released.pth` 
```
Similarly, you can specify `algo` as you like.


## Saving and Loading Checkpoints

The agent periodically saves the model's weights and the target network state for later resumption. 

```python
def save_checkpoint(self, file_path):
    checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'target_model_state_dict': self.target_model.state_dict()
    }
    torch.save(checkpoint, file_path)
```
You can load a checkpoint by setting the `--load` flag and choosing `logs/{task}_{algo}_checkpoint_released.pth` (if it has been saved).

## MacOS Usage

- Training

You can add `-d mps` to train:
```bash
python run_ppo_vision.py -n 10 -d mps
```

- Evaluation

You can add `-d mps` to eval and visualization:
```bash
python run_ppo_vision.py -l -v -n 1 -t GraspFixedBlock -d mps
```
