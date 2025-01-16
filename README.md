# Xu Master Thesis

# Development Log
Jiacheng Xu @ KTH

### Usage:
D435 camera capture keypoints for ABB, UR Robot.

# 修改的文件
## 1. 增加了real_cam_utils.py，用于获取真实摄像头数据,代替og_utils.py.
## 2. 修改了main.py，用于使用真实摄像头数据
## 跑以下命令，使用真实摄像头数据
'''bash
python main.py --use_real_camera --visualize
'''


# 40 天完成会议论文：任务清单 (Task List)


---

## 第 0-5 天：关键点捕捉

- [ ] **整理用于训练的数据集** 
    - [ ] Pouring tea
    - [x] Put the green package in the drawer
    - [x] Close the drawer
    - [x] Open the drawer
    - [x] Put down the green package
- [ ] **跑通实体摄像头关键点捕捉**
---

## 第 6-10 天：DinoX 初步实现

- [ ] **环境与依赖安装**  
  - [ ] 拉取 DinoX 源码或预训练模型，解决环境依赖问题  
  - [ ] 测试最小可运行脚本，验证模型能正常推理

- [ ] **在简单场景中测试 DinoX**  
  - [ ] 准备一小批图像或视频，运行 DinoX 检测  
  - [ ] 观察检测精度与模型表现

- [ ] **评估检测结果与关键点**  
  - [ ] 将检测结果与关键点捕捉模块做简单对比 / 可视化  
  - [ ] 讨论如何从检测框或掩码中提取需要的关键点信息

---

## 第 11-15 天：Gaussian Splatting（3D 重建）初步实现

- [ ] **环境搭建与示例运行**  
  - [ ] 确定 3D 重建工具链 (官方 C++/Python 接口、或 ROS 集成)  
  - [ ] 跑通 Gaussian Splatting 的官方 Demo 或开源示例

- [ ] **与相机数据结合**  
  - [ ] 捕捉多视角图像/视频，输入 Gaussian Splatting 管线  
  - [ ] 评估点云或 3D 表示的效果与质量

- [ ] **与 DinoX 简单对接**  
  - [ ] 尝试在多视角下，将 DinoX 检测到的目标信息投影到 3D 重建结果中  
  - [ ] 确定 2D→3D 映射策略，形成初步融合思路

---

## 第 16-20 天：ReKep（任务分解与优化）基础实现

- [ ] **核心论文与概念实现**  
  - [ ] 通读 ReKep 论文，理解 Relational Keypoint Constraints (RKC) 原理  
  - [ ] 如果有开源代码，尝试在示例数据上跑通优化流程

- [ ] **示例场景搭建**  
  - [ ] 选取简单的 pick-and-place 任务（如抓取杯子）  
  - [ ] 使用 ReKep 的约束表示与优化流程，获得机器人运动轨迹

- [ ] **与检测/重建信息融合**  
  - [ ] 读取 DinoX + Gaussian Splatting 的部分输出，传入 ReKep  
  - [ ] 完成一个最小可行的任务规划 Demo

---

## 第 21-25 天：整体集成与初步实验

- [ ] **端到端管线搭建**  
  - [ ] 摄像头采集图像 → DinoX 检测 → Gaussian Splatting 3D 重建 → ReKep 任务规划  
  - [ ] 输出可行的抓取或移动指令 (可在仿真或真实机器人上测试)

- [ ] **多次实验与调试**  
  - [ ] 针对同一个简单场景重复测试，并记录系统稳定性  
  - [ ] 收集定量指标（检测准确率 / 重建精度 / 成功抓取率等）

- [ ] **瓶颈排查与性能优化**  
  - [ ] 针对检测、重建、优化各模块做超参数或工程优化  
  - [ ] 记录并分析主要的性能瓶颈

---

## 第 26-30 天：扩展场景与论文主体写作

- [ ] **更多测试场景**  
  - [ ] 增加不同物体或更复杂的场景进行测试  
  - [ ] 收集更多实验结果，用于论文撰写

- [ ] **撰写论文初稿结构**  
  - [ ] 完成大纲：引言、相关工作、方法、实验、结论  
  - [ ] 将现有实验结果整理到论文，配合可视化图表

- [ ] **初步结果可视化**  
  - [ ] 展示 DinoX 检测、Gaussian Splatting 重建、ReKep 轨迹生成等可视化截图  
  - [ ] 确保论文结构清晰，让读者能理解端到端流程

---

## 第 31-35 天：打磨实验与论文细节

- [ ] **补充对比 / 消融实验**  
  - [ ] 对比其他检测器 / 重建方法 / 优化方式 (如无 DinoX / 传统 SfM 等)  
  - [ ] 做消融：移除某个模块，看性能如何变化

- [ ] **失败案例与局限性分析**  
  - [ ] 收集在光照不佳、强遮挡或动态场景下的失败案例  
  - [ ] 在论文中总结局限性与改进方向

- [ ] **完善论文行文与逻辑**  
  - [ ] 突出贡献点与创新之处  
  - [ ] 进一步优化语言与排版，减少冗余

---

## 第 36-40 天：最终实验整理与论文定稿

- [ ] **确认最终实验结果**  
  - [ ] 复查所有表格、图示及指标，避免出现错误数据  
  - [ ] 如有需要，补做小规模重复实验

- [ ] **撰写摘要、引言与结论**  
  - [ ] 精炼提炼本文亮点，总结贡献  
  - [ ] 突出 DinoX、Gaussian Splatting、ReKep 整合后的提升

- [ ] **检查投稿格式与排版**  
  - [ ] 按会议要求调整页数、引用格式、排版细节  
  - [ ] 做查重（若需），并确保论文合规

- [ ] **提交与后续安排**  
  - [ ] 在会议系统完成注册与论文上传  
  - [ ] 规划后续补充实验 / 答辩等事宜

---




## ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation

#### [[Project Page]](https://rekep-robot.github.io/) [[Paper]](https://rekep-robot.github.io/rekep.pdf) [[Video]](https://youtu.be/2S8YhBdLdww)


## Setup Instructions

Note that this codebase is best run with a display. For running in headless mode, refer to the [instructions in OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html).

- Install [OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html). This code is tested on [this commit](https://github.com/StanfordVL/OmniGibson/tree/cc0316a0574018a3cb2956fcbff3be75c07cdf0f).

NOTE: If you encounter the warning `We did not find Isaac Sim under ~/.local/share/ov/pkg.` when running `./scripts/setup.sh` for OmniGibson, first ensure that you have installed Isaac Sim. Assuming Isaac Sim is installed in the default directory, then provide the following path `/home/[USERNAME]/.local/share/ov/pkg/isaac-sim-2023.1.1` (replace `[USERNAME]` with your username).

- Install ReKep in the same conda environment:
```Shell
conda activate omnigibson
cd ..
git clone https://github.com/huangwl18/ReKep.git
cd ReKep
pip install -r requirements.txt
```

- Obtain an [OpenAI API](https://openai.com/blog/openai-api) key and set it up as an environment variable:
```Shell
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## Running Demo

We provide a demo "pen-in-holder" task that illustrates the core idea in ReKep. Below we provide several options to run the demo.

Notes:
- An additional `--visualize` flag may be added to visualize every solution from optimization, but since the pipeline needs to repeatedly solves optimization problems, the visualization is blocking and needs to be closed every time in order to continue (by pressing "ESC").
- Due to challenges of determinism of the physics simulator, different runs with the same random seed may produce different results. It is possible that the robot may fail at the provided task, especially when external disturbances are applied. In this case, we recommend running the demo again.

### Demo with Cached Query

We recommend starting with the cached VLM query.

```Shell
python main.py --use_cached_query [--visualize]
```

A video will be saved to `./videos` by default.

<img  src="media/pen-in-holder.gif" width="480">

### Demo with External Disturbances

Since ReKep acts as a closed-loop policy, it is robust to disturbances with automatic failure recovery both within stages and across stages. To demonstrate this in simulation, we apply the following disturbances for the "pen-in-holder" task:

- Move the pen when robot is trying to grasp the pen

- Take the pen out of the gripper when robot is trying to reorient the pen

- Move the holder when robot is trying to drop the pen into the holder

<img  src="media/pen-in-holder-disturbances.gif" width="480">

Note that since the disturbances are pre-defined, we recommend running with the cached query.

```Shell
python main.py --use_cached_query --apply_disturbance [--visualize]
```
### Demo with Live Query

The following script can be run to query VLM for a new sequence of ReKep constraints and executes them on the robot:

```Shell
python main.py [--visualize]
```

## Setup New Environments
Leveraging the diverse objects and scenes provided by [BEHAVIOR-1K](https://behavior.stanford.edu/) in [OmniGibson](https://behavior.stanford.edu/omnigibson/index.html), new tasks and scenes can be easily configured. To change the objects, you may check out the available objects as part of the BEHAVIOR assets on this [page](https://behavior.stanford.edu/knowledgebase/objects/index.html) (click on each object instance to view its visualization). After identifying the objects, we recommend making a copy of the JSON scene file `./configs/og_scene_file_pen.json` and edit the `state` and `objects_info` accordingly. Remember that the scene file need to be supplied to the `Main` class at initialization. Additional [scenes](https://behavior.stanford.edu/knowledgebase/scenes/index.html) and [robots](https://behavior.stanford.edu/omnigibson/getting_started/examples.html#robots) provided by BEHAVIOR-1K may also be possible, but they are currently untested.

## Real-World Deployment
To deploy ReKep in the real world, most changes should only be needed inside `environment.py`. Specifically, all of the "exposed functions" need to be changed for the real world environment. The following components need to be implemented:

- **Robot Controller**: Our real-world implementation uses the joint impedance controller from [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control) for our Franka setup. Specifically, when `execute_action` in `environment.py` receives a target end-effector pose, we first calculate IK to obtain the target joint positions and send the command to the low-level controller.
- **Keypoint Tracker**: Keypoints need to be tracked in order to perform closed-loop replanning, and this typically needs to be achieved using RGD-D cameras. Our real-world implementation uses similarity matching of [DINOv2](https://github.com/facebookresearch/dinov2) features calculated from multiple RGB-D cameras to track the keypoints (details may be found in the [paper](https://rekep-robot.github.io/rekep.pdf) appendix). Alternatively, we also recommend trying out specialized point trackers, such as [\[1\]](https://github.com/henry123-boy/SpaTracker), [\[2\]](https://github.com/google-deepmind/tapnet), [\[3\]](https://github.com/facebookresearch/co-tracker), and [\[4\]](https://github.com/aharley/pips2).
- **SDF Reconstruction**: In order to avoid collision with irrelevant objects or the table, an SDF voxel grid of the environment needs to be provided to the solvers. Additionally, the SDF should ignore robot arm and any grasped objects. Our real-world implementation uses [nvblox_torch](https://github.com/NVlabs/nvblox_torch) for ESDF reconstruction, [cuRobo](https://github.com/NVlabs/curobo) for segmenting robot arm, and [Cutie](https://github.com/hkchengrex/Cutie) for object mask tracking.
- **(Optional) Consistency Cost**: If closed-loop replanning is desired, we find it helpful to include a consistency cost in the solver to encourage the new solution to be close to the previous one (more details can be found in the [paper](https://rekep-robot.github.io/rekep.pdf) appendix).
- **(Optional) Grasp Metric or Grasp Detector**: We include a cost that encourages top-down grasp pose in this codebase, in addition to the collision avoidance cost and the ReKep constraint (for identifying grasp keypoint), which collectively identify the 6 DoF grasp pose. Alternatively, other grasp metrics can be included, such as force-closure. Our real-world implementation instead uses grasp detectors [AnyGrasp](https://github.com/graspnet/anygrasp_sdk), which is implemented as a special routine because it is too slow to be used as an optimizable cost.

Since there are several components in the pipeline, running them sequentially in the real world may be too slow. As a result, we recommend running the following compute-intensive components in separate processes in addition to the main process that runs `main.py`: `subgoal_solver`, `path_solver`, `keypoint_tracker`, `sdf_reconstruction`, `mask_tracker`, and `grasp_detector` (if used).

## Known Limitations
- **Prompt Tuning**: Since ReKep relies on VLMs to generate code-based constraints to solve for the behaviors of the robot, it is sensitive to the specific VLM used and the prompts given to the VLM. Although visual prompting is used, typically we find that the prompts do not necessarily need to contain image-text examples or code examples, and pure-text high-level instructions can go a long way with the latest VLM such as `GPT-4o`. As a result, when starting with a new domain and if you observe that the default prompt is failing, we recommend the following steps: 1) pick a few representative tasks in the domain for validation purposes, 2) procedurally update the prompt with high-level text examples and instructions, and 3) test the prompt by checking the text output and return to step 2 if needed.

- **Performance Tuning**: For clarity purpose, the entire pipeline is run sequentially. The latency introduced by the simulator and the solvers gets compounded. If this is a concern, we recommend running compute-intensive components, such as the simulator, the `subgoal_solver`, and the `path_solver`, in separate processes, but concurrency needs to be handled with care. More discussion can be found in the "Real-World Deployment" section. To tune the solver, the `objective` function typically takes the majority of time, and among the different costs, the reachability cost by the IK solver is typically most expensive to compute. Depending on the task, you may reduce `sampling_maxfun` and `maxiter` in `configs/config.yaml` or disable the reachability cost. 

- **Task-Space Planning**: Since the current pipeline performs planning in the task space (i.e., solving for end-effector poses) instead of the joint space, it occasionally may produce actions kinematically challenging for robots to achieve, especially for tasks that require 6 DoF motions.

## Troubleshooting

For issues related to OmniGibson, please raise a issue [here](https://github.com/StanfordVL/OmniGibson/issues). You are also welcome to join the [Discord](https://discord.com/invite/bccR5vGFEx) channel for timely support.

For other issues related to the code in this repo, feel free to raise an issue in this repo and we will try to address it when available.