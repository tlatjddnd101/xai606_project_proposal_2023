# XAI606 project proposal 2023
The baseline code for XAI606 project proposal, 2023

## Project Overview
The objective of this project is to predict whether the reservation was cancelled or not. The train dataset and test dataset are given with .csv file in the *datasets* directory. The target of prediction is `booking_status`, which means whether the reservation was cancelled. Train your model with **train.csv** and predict the `booking_status` of **test.csv**!

## Recommened Docker Image
To run this code, we need JAX and PyTorch. The following docker image have both.
```bash
docker pull tlatjddnd101/base:jpt_mujoco
```

## How to run
Set the target GPU to be used and random seed before run.
```bash
./run.sh
```

## Evaluation
If you want to evaluate your model's prediction, go on [original project webise](https://www.kaggle.com/competitions/playground-series-s3e7) and submit the **prediction.csv** file in the generated `run_log` directory.