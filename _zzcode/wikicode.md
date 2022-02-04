---
title: Code wiki
author: Anass El Aouni
date: 2022-02-03
category: mldroppage
layout: post
---


Before starting, you will need to install the following packages, preferably on conda:

>[Ocean Parcel](https://oceanparcels.org)

>[Xarray](https://xarray.pydata.org/en/stable/)

>[Pytorch](https://pytorch.org/)

>[Torchvision](https://github.com/pytorch/vision)

>[netcdf4](https://unidata.github.io/netcdf4-python/)

***

The tree of the project looks like:

```
Project
|
├── config
│   └── data.py
├── core
│   ├── data
│   │   ├── datasets.py
│   │   ├── generate.py
│   │   └── process.py
│   ├── model
│   │   ├── __init__.py
│   │   └── UNet
│   │       ├── model.py
│   │       └── parts.py
│   ├── simulate
│   │   ├── __init__.py
│   │   └── trajectories
│   │       ├── io.py
│   │       ├── kernels.py
│   │       └── particles.py
│   └── util
│       ├── geodesic.py
│       ├── grid.py
│       ├── __init__.py
│       ├── misc.py
│       ├── path.py
│       └── random.py
├── density_maps.py
├── load.py
├── misc.py
├── README.md
├── snapshot.py
└── test_model.py
```



***

+ Use ```python density_maps.py``` to generate input data for training:

```
training data
|
├── density_maps
├── index_offsets.npy
├── initial_positions
├── tracers_fields
├── trajectories
└── velocity_fields

```

+ Edit  ```misc.py``` to configure the input data, train/eval, directories..  and run ``` python misc.py```.

+ Edit ``` snapshot.py``` and run ```python snapshot "name_of_model"``` to start learning.

