# Xu Master Thesis

# Development Log
Jiacheng Xu @ KTH

Reproduce [Rekep](https://arxiv.org/abs/2409.01652) in real robotic arm (UR5) and one camera (D435)

# Actual Operating (locked end-effector pose)
<img  src="media/pen-in-holder.gif" width="480">

# Steps Breakingdown

## Environment Setup
```bash
Ubuntu20 or Ubuntu22
Cuda: 12.4
```
[hand-on-eye Camera Calibration](https://github.com/heyjiacheng/hand-eye-calibration) (Optional)

Obtain an [OpenAI API](https://openai.com/blog/openai-api) key and set it up as an environment variable:
```Shell
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```
Obtain an [Dino-X API](https://cloud.deepdataspace.com/playground/dino-x?referring_prompt=0) key and set it up as an environment variable:
```Shell
export Dino_X_API_KEY="YOUR_Dino_X_API_KEY"
```

## Installation

1. First, install [pixi](https://pixi.sh/latest/#installation)

2. Then build the dependencies:
```bash
pixi install
```

## Take photo（use D435, available at ./data/realsense_captures）
```bash
python real_camera.py
```

## Run [Dino-X](https://arxiv.org/abs/2411.14347) model to do image segementation. Clustering and flitering keypoints from 2D mask (generatede by Dino-X)
```bash
python real_vision.py
```

## Transfer keypoint position from camera coordinate to world coordinate (robot base coordinate). Generate robotic arm action, based on optimizer (Dual Annealing and SLSQP)
```bash
python real_action.py
```

## Visualization robot coordinate and sequence of end-effector trajectory.
```bash
python visualization.py
```

## Conduct action on UR5 robotic arm
```bash
python ur5_action.py
```



# Original Work

## ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation

#### [[Project Page]](https://rekep-robot.github.io/) [[Paper]](https://rekep-robot.github.io/rekep.pdf) [[Video]](https://youtu.be/2S8YhBdLdww)

