---
title: Weekly discussion (March 07)
author: Anass El Aouni
date: 2022-02-04
category: mldroppage
layout: post
---



-------

# Notes from last meeting

* [x] Normalize maps between 0~1
* [x] Compare to Unet (same config 100 samples)
* [x] Increase to x5 with ResNet
* [ ] Discuss new norms
* [x] Why initial batch errors are not the same everywhere? Weights are initialized randomly each time)
* [x] Where are compared to Toulon config?: 10.000 samples of 30-day trajectories (Number of simple is similar to what we had before)

-------


# Simulation 1


###### Configuration of Simu 1:

We reduced the number of data to:

```yaml
    n_samples : 100
    ensemble_size : 2000
    ensemble_radius : 5 km
    particle_runtime : 3 days
    particle_output_dt : 6 hours
    particle_dt_RK4 : dt(hour:=3)
```

1. Simulation 3ch 
   1. Density maps are normalized between 0 and 1
   2. In addition to the density maps, the machine takes u and v as input
2. Simulation 5ch
   1. Density maps are multiplied by dxdy and later normalized between 0 and 1
   2. In addition to the density maps, u and v, the machine takes dx and dy as input
   3. dx an dxy are normalized between 0 and 1
   

###### Architectures of Simu 1:
In this simulation we use both RESnet and Unet architectures:

```
RESnet
+-------------------------------------+------------+
|               Modules               | Parameters |
+-------------------------------------+------------+
|         initial_layer.weight        |     80     |
|          initial_layer.bias         |     16     |
| blocks.0.batch_norm_layers.0.weight |     16     |
|  blocks.0.batch_norm_layers.0.bias  |     16     |
| blocks.0.batch_norm_layers.1.weight |     16     |
|  blocks.0.batch_norm_layers.1.bias  |     16     |
|    blocks.0.conv_layers.0.weight    |    2304    |
|     blocks.0.conv_layers.0.bias     |     16     |
|    blocks.0.conv_layers.1.weight    |    2304    |
|     blocks.0.conv_layers.1.bias     |     16     |
|          final_layer.weight         |     16     |
|           final_layer.bias          |     1      |
+-------------------------------------+------------+
Total Trainable Params: 4817
```

```
Unet
+-------------------------------------------+------------+
|                  Modules                  | Parameters |
+-------------------------------------------+------------+
|          inc.double_conv.0.weight         |    1728    |
|           inc.double_conv.0.bias          |     64     |
|          inc.double_conv.1.weight         |     64     |
|           inc.double_conv.1.bias          |     64     |
|          inc.double_conv.3.weight         |   36864    |
|           inc.double_conv.3.bias          |     64     |
|          inc.double_conv.4.weight         |     64     |
|           inc.double_conv.4.bias          |     64     |
| down1.maxpool_conv.1.double_conv.0.weight |   73728    |
|  down1.maxpool_conv.1.double_conv.0.bias  |    128     |
| down1.maxpool_conv.1.double_conv.1.weight |    128     |
|  down1.maxpool_conv.1.double_conv.1.bias  |    128     |
| down1.maxpool_conv.1.double_conv.3.weight |   147456   |
|  down1.maxpool_conv.1.double_conv.3.bias  |    128     |
| down1.maxpool_conv.1.double_conv.4.weight |    128     |
|  down1.maxpool_conv.1.double_conv.4.bias  |    128     |
| down2.maxpool_conv.1.double_conv.0.weight |   294912   |
|  down2.maxpool_conv.1.double_conv.0.bias  |    256     |
| down2.maxpool_conv.1.double_conv.1.weight |    256     |
|  down2.maxpool_conv.1.double_conv.1.bias  |    256     |
| down2.maxpool_conv.1.double_conv.3.weight |   589824   |
|  down2.maxpool_conv.1.double_conv.3.bias  |    256     |
| down2.maxpool_conv.1.double_conv.4.weight |    256     |
|  down2.maxpool_conv.1.double_conv.4.bias  |    256     |
| down3.maxpool_conv.1.double_conv.0.weight |  1179648   |
|  down3.maxpool_conv.1.double_conv.0.bias  |    512     |
| down3.maxpool_conv.1.double_conv.1.weight |    512     |
|  down3.maxpool_conv.1.double_conv.1.bias  |    512     |
| down3.maxpool_conv.1.double_conv.3.weight |  2359296   |
|  down3.maxpool_conv.1.double_conv.3.bias  |    512     |
| down3.maxpool_conv.1.double_conv.4.weight |    512     |
|  down3.maxpool_conv.1.double_conv.4.bias  |    512     |
| down4.maxpool_conv.1.double_conv.0.weight |  2359296   |
|  down4.maxpool_conv.1.double_conv.0.bias  |    512     |
| down4.maxpool_conv.1.double_conv.1.weight |    512     |
|  down4.maxpool_conv.1.double_conv.1.bias  |    512     |
| down4.maxpool_conv.1.double_conv.3.weight |  2359296   |
|  down4.maxpool_conv.1.double_conv.3.bias  |    512     |
| down4.maxpool_conv.1.double_conv.4.weight |    512     |
|  down4.maxpool_conv.1.double_conv.4.bias  |    512     |
|       up1.conv.double_conv.0.weight       |  4718592   |
|        up1.conv.double_conv.0.bias        |    512     |
|       up1.conv.double_conv.1.weight       |    512     |
|        up1.conv.double_conv.1.bias        |    512     |
|       up1.conv.double_conv.3.weight       |  1179648   |
|        up1.conv.double_conv.3.bias        |    256     |
|       up1.conv.double_conv.4.weight       |    256     |
|        up1.conv.double_conv.4.bias        |    256     |
|       up2.conv.double_conv.0.weight       |  1179648   |
|        up2.conv.double_conv.0.bias        |    256     |
|       up2.conv.double_conv.1.weight       |    256     |
|        up2.conv.double_conv.1.bias        |    256     |
|       up2.conv.double_conv.3.weight       |   294912   |
|        up2.conv.double_conv.3.bias        |    128     |
|       up2.conv.double_conv.4.weight       |    128     |
|        up2.conv.double_conv.4.bias        |    128     |
|       up3.conv.double_conv.0.weight       |   294912   |
|        up3.conv.double_conv.0.bias        |    128     |
|       up3.conv.double_conv.1.weight       |    128     |
|        up3.conv.double_conv.1.bias        |    128     |
|       up3.conv.double_conv.3.weight       |   73728    |
|        up3.conv.double_conv.3.bias        |     64     |
|       up3.conv.double_conv.4.weight       |     64     |
|        up3.conv.double_conv.4.bias        |     64     |
|       up4.conv.double_conv.0.weight       |   73728    |
|        up4.conv.double_conv.0.bias        |     64     |
|       up4.conv.double_conv.1.weight       |     64     |
|        up4.conv.double_conv.1.bias        |     64     |
|       up4.conv.double_conv.3.weight       |   36864    |
|        up4.conv.double_conv.3.bias        |     64     |
|       up4.conv.double_conv.4.weight       |     64     |
|        up4.conv.double_conv.4.bias        |     64     |
|              outc.conv.weight             |     64     |
|               outc.conv.bias              |     1      |
+-------------------------------------------+------------+
Total Trainable Params: 17267393

```


###### Results of Simu 1:

Comparisons between Epochs
![ocean current](../../assets/images/resunet100_1.png)
Comparisons between batches
![ocean current](../../assets/images/resunet100_2.png)



-------

# Simulation 2


###### Configuration of Simu 2:

We reduced the number of data to:

```yaml
    n_samples : 500
    ensemble_size : 2000
    ensemble_radius : 5 km
    particle_runtime : 3 days
    particle_output_dt : 6 hours
    particle_dt_RK4 : dt(hour:=3)
```

1. Simulation 3ch 
   1. Density maps are normalized between 0 and 1
   2. In addition to the density maps, the machine takes u and v as input
2. Simulation 5ch
   1. Density maps are multiplied by dxdy and later normalized between 0 and 1
   2. In addition to the density maps, u and v, the machine takes dx and dy as input
   3. dx an dxy are normalized between 0 and 1
   

###### Architectures of Simu 2:

The RESnet described above

###### Results of Simu 2:

Comparisons between Epochs
![ocean current](../../assets/images/res500_1.png)
Comparisons between batches
![ocean current](../../assets/images/res500_2.png)


-------

###### Comparison between all simulations above

![ocean current](../../assets/images/com_resnet_all.png)

###### Your comments are welcomed here
{% include comments.html %}
