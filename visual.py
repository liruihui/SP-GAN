#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: train.py 
@time: 2019/09/17
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""
import os
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime

from Generation.model_test import Model
from Generation.config import opts

if __name__ == '__main__':


    opts.pretrain_model_G = "table_G.pth"
    opts.log_dir = "models"

    model = Model(opts)

    model.draw_correspondense() # draw the correspondense between sphere and shape
    #model.draw_shape_intepolate() # shape inteporlate
    #model.draw_part_shape_inte()  # shape inteporlate vs part-wise shape inteporlate
    #model.draw_part_shape_inte_detail() # shape inteporlate vs multi-path part-wise shape inteporlate

    #model.draw_part_edit() # random change the noise on selected region
    #model.draw_part_flip() # negative the noise vector along x,y,z zxis
    #model.draw_edit_inte() # combine for part edit & part/shape interpolate

    #model.draw_part_exchange() # exchange the noise vector of two regions of two shape



