# XAI606 project proposal 2023
The baseline code for XAI606 project proposal, 2023

## Project Overview
This project is to predict whether the reservation was cancelled or not. The train dataset and test dataset are given with .csv file in the *datasets* directory. The target of prediction is `booking_status`, which means whether the reservation was cancelled. Train your model with **train.csv** and predict the `booking_status` of **test.csv**!

## Docker Image to use
```bash
docker pull tlatjddnd101/base:jpt_mujoco
```

## How to run
```bash
./run.sh
```