# Map Space Belief Prediction for<br/> Manipulation-Enhanced Mapping
![Overview figure](./shelf_gym/images/overview.png)
This repository contains the code for reproducing the work of the RSS 2025 paper ["Map Space Belief Prediction for Manipulation-Enhanced Mapping"](https://arxiv.org/pdf/2502.20606), as well as the code for training the Calibrated Neural-Accelerated
Belief Update (CNABU) networks introduced in this paper. 

**This repository is a work in progress and currently contains the code required to run the proposed overall pipeline, as well as the data collection for network training.
The specific network training code and evaluation code used for the paper will be uploaded and updated in the near future.**

# Overview
![Overview figure](./shelf_gym/images/architecture.png)
From a prior map belief, our pipeline predicts a map belief resulting from a set of candidate pushes. It then weighs the information gain from taking two consecutive independent views given the current belief (orange arrows) or taking a single observation given any of the predicted beliefs after pushing (blue arrows), selecting the path of highest cumulative information gain and taking its respective first action -- either taking the next best view or executing the best push. $IGV_t$ represents the best information gain obtainable from taking two distinct observation actions, whereas $IGM_t$ is the best information gain obtainable through a manipulation action followed by an observation action.


# Installation
In order to download the pre-trained CNABU models, please install ```git-lfs``` (apt install git-lfs).
To install and use the gym environment for this project we suggest using Anaconda3 or other virtual environments.
The code is tested with python >= 3.9 & <=3.12. 

- First, clone the repository including the submodule for scikit-geometry, as it is needed to run the code:
```bash
  git clone --recurse-submodules -j8 https://github.com/NilsDengler/manipulation_enhanced_map_prediction
  ```

- To install all the necessary packages, start your Conda environment run:
```bash
  conda activate YOUR_ENV_NAME
  cd YOUR_INSTALLATION_PATH/manipulation_enhanced_map_prediction
  git submodule update --init --recursive
  git submodule update --recursive --remote
  ./install.sh
  ```

# Structure
The project is structured as follows:
- ```shelf_gym``` contains the whole simulation structure.
  - ```shelf_gym/environments``` contains:
    - The base Pybullet environment ```base_environment.py```
    - The general generation of an ur5 robot in pybullet ```ur5_environment.py```
    - The world building script for the specific shelf environment. ```shelf_environment.py```
  - ```shelf_gym/meshes``` contains the meshes for the ur5, robotiq 85f2 gripper, YCB objects and environment specifics.
  - ```shelf_gym/scripts``` contains specific task related code which goes beyond the environment building.
    - ```shelf_gym/scripts/model``` contains the pre-tained CNABU-models, used fixed camera array, and a demo dataset used for utility functions
    - ```shelf_gym/scripts/data_generation``` contains the files to generate data for mapping and pushing to train the CNABUS
    - ```shelf_gym/scripts/model_training``` contains the files to train the CNABU models (**NOT updated yet**)
    - The **full pipeline script** ```run_cnabu_pipeline.py```, run this to replicate the papers results

  - ```shelf_gym/utils``` contains the utilities for the environment.

  
# Base Demo
To run a demo of the environment without executing the pipeline you can use the following command:
```bash 
  python shelf_gym/environments/shelf_environment.py
```

# Manipulation-Enhanced Mapping Demo
To run the demo of the full manipulation-enhanced mapping pipeline, as proposed in the paper, you can use the following command:
```bash 
  cd shelf_gym/scripts
  python run_cnabu_pipeline.py
```

# Data Collection Demo
To collect mapping or pushing data to train the CNABUS, run the following code:
```bash 
  cd shelf_gym/scripts/data_generation
  python map_collection.py #for map data only
  python push_collection.py #for map and push data pre- and post-psuh
```
Alternatively, you can collect the data for the [viewpoint push planning](https://github.com/NilsDengler/view-point-pushing) work by Dengler et al. using the following method:
```bash 
  cd shelf_gym/scripts/data_generation
  python dengler_iros_2023_map_collection.py 
```

# Issue Tracker
 - 

