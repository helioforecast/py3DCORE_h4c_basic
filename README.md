## 3DCORE_h4c extensions (ver 0.0.0)
========

Extensions to the 3D Coronal Rope Ejection modelling techniqe for coronal mass ejection (CME) flux ropes. The basic version can be found in the repository 'py3DCORE_h4c_basic'. 

# Installation
------------

Install the latest version manually using `git`:

    git clone https://github.com/helioforecast/py3DCORE_h4c
    cd py3DCORE_h4c
    pip install .
    pip install -r py3dcore_h4c_requirements.txt

or the original version from https://github.com/ajefweiss/py3DCORE.

------------

# Notes on heliosat
------------

In order for heliosat to work properly, the following steps are necessary:

1. manually create the folder .heliosat 
2. within .heliosat, manually create the following three folders
    - cache
    - data
    - kernels
