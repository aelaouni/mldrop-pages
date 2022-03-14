---
title: Weekly discussion (March 14)
author: Anass El Aouni
date: 2022-03-14
category: mldroppage
layout: post
---



-------

# Notes from last meeting

* [x] check errors within density maps 
* [x] Run same simulation 4~5 times to see initial errors
* [x] Set learning curves in percentage 

-------


# Check error within density maps

We star first by choosing a fixed location X0 south of the islands, and a fixed date t0. Then we generate 500 samples with a size of 2000 floats each time in a box of 5km centered by X0. The density maps are normalized between [0, 1], and they are all advected between t0 and t0+3days.
We measure the errors of each sample as the average deviation of the density maps from the mean average over the 500 samples (for non-zeros):

``` python

for i in range(500): #loop over samples 
    for t in range(12): # loop over time steps
       error[i,t]= nonzero_mean(sample[i,t]- mean(samples[:,t]))/nonzero_mean(mean(samples[:,t]))
       
```
![ocean current](../../assets/images/hov_err.png)


# Run same simulation 4~5 times to see initial errors

###### Resnet 3ch
![ocean current](../../assets/images/res3.png)

###### Resnet 5ch
![ocean current](../../assets/images/res5.png)

# Set learning curves in percentage 

![ocean current](../../assets/images/loss_per.png)

###### Your comments are welcomed here
{% include comments.html %}
