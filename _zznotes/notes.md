---
title: Notes
author: Anass El Aouni
date: 2022-02-03
category: mldroppage
layout: post
---



-------

# Unet Architecture
The architecture we are using is the following:

```python
import torch.nn as nn

from core.model.UNet.parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```

With``` n_channels=3, n_classes=1```, a visualization of this architecture is shown in the following figure
![ocean current](../../assets/images/Unet_arch.png)


+ I quote from [[1]](https://arxiv.org/pdf/1505.04597v1.pdf) **"_We modify and extend this architecture such that it works with very few training images and yields more precise segmentations_"**.
This is one of the reasons to explain the fast learning rate we have in our simulations.

+ Over 100.000 images to learn from is too much (After discussing with ML guys: normal to Learn all from first the epoch).

+ From the same paper I quote **"_The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization_"**. This architecture does take spatial information! (confirmed by J-E.J).


-------
# Number of parameters

```
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


-------

# Back to data

Averaging all the 10.000x13 images of density maps look like this:

![ocean current](../../assets/images/meandens.png)

+ This could also be one of the reason to converge quite fast. Although we have quite a large domain, most of the actions do happen in a very specific area!

Also we can evaluate these densities at each time step of the 3-days advection, the following video shows the average of the 10.000 samples at each 6-hours time step.
![ocean current](../../assets/images/mdens.gif)


My takes from this are:

+ Looks like we are only learning diffusion-like advection  
+ Maybe we can use backward advection lo lean convergence?


-------
# Changing arch or input data?

I played a bit with the data and generated few input of converging flows to check whether we learn this information of not:

![ocean current](../../assets/images/reverse.gif)

Two things to take from this experiments:

1. Learning position is more or less fine in both direction of time.
2. Intensity is not that good, we learn more about divergence than convergence. (--> add conservation of )




###### Ideas on making more diverse data



- Generate few data along attracting and repelling LCSs to learn strong convergence/divergence:

<img src="../../assets/images/lcs.jpeg" alt="drawing" width="500"/>

- Also initialize densities along hyperbolic points, such points correspond to areas with strong mixing activity: fluid is advected here along a compression line and then dispersed away along the stretching line:

<img src="../../assets/images/hyppoint.png" alt="drawing" width="200"/>

- Maybe withing coherent eddies (where the densities should remain more or less the same).
- Maybe sink flow, add W? 

###### Architecture

Maybe a ```SingleConv``` instead of ```DoubleConv``` in case we keep Unet.


-------



[[1]](https://arxiv.org/pdf/1505.04597v1.pdf) Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.




------

###### Your comments are welcomed here
{% include comments.html %}
