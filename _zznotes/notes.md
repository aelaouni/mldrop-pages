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

With``` n_channels=3, n_classes=1```, a simplification of this architecture is shown in the following figure
![ocean current](../../assets/images/Unet_arch.png)


+ I quote from [[1]](https://arxiv.org/pdf/1505.04597v1.pdf) **"_We modify and extend this architecture such that it works with very few training images and yields more precise segmentations_"**.
This is one of the reasons to explain the fast learning rate we have in our simulations.

+ Over 100.000 images to learn from is too much (After discussing with ML guys: normal to Learn all from first the epoch).

+ From the same paper I quote **"_The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization_"**. This architecture does take spatial information! (confirmed by J-E.J).


-------

# Back to data

Averaging all the 10.000 samples of density maps look like this:

![ocean current](../../assets/images/meandens.png)

+ This could also be one of the reason to converge quite fast. Although we have quite a large domain, most of the actions do happen in a very specific area!

-------



[[1]](https://arxiv.org/pdf/1505.04597v1.pdf) Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.




------

###### Your comments are welcomed here:
{% include comments.html %}
