#!/bin/bash

#pip install git+git://github.com/erikwijmans/etw_pytorch_utils.git@v1.1.1#egg=etw_pytorch_utils
#pip install --no-cache --upgrade git+https://github.com/dongzhuoyao/pytorchgo.git

cd emd_ && python setup.py build_ext --inplace && cd ..
cd cd && python setup.py build_ext --inplace && cd ..

