# Copyright (c) Phigent Robotics. All rights reserved.
_base_ = ['./bevdet_waymo.py']

fp16 = dict(loss_scale='dynamic')