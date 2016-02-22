# TranSpike

This is my first attempt to build a transformer for Spiking Neural Networks.
The idea is roughly following the paper *Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing*.

This project is purely built by using Torch 7.

## Updates

+ Load pretrained VGG-16 network or TeraDeep Network [2016-02-09]
+ Print VGG-16 model, filters, and input, output stream [2016-02-09]
+ Write a pre-mature version of Spiking ReLU class [2016-02-10]
+ Setup experiments with single image to get reasonable results [TODO]
+ Setup experiments with multiple images to get reasonable results [TODO]
+ Find a way of setting up a proper classification experiment (MNIST with LeNet now) [2016-02-12]
+ Figure out a way of hooking this modified network with a camera [TODO]
+ Setup experiments as indicated in Python code [2016-02-22]

## Notes

### Network Structure of TeraDeep's 1000 Categories Net

   Designed architecture should receive 224x224 input, however, the architecture can still receive images from 221x221 to 252x252.
   The height and width doesn't have to be same in this range.

   ```
   nn.Sequential {
  [input(224x224) -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> output]
  (1): nn.SpatialConvolutionMM(3 -> 96, 9x9, 4,4)
  (2): nn.ReLU
  (3): nn.SpatialConvolutionMM(96 -> 256, 5x5, 1,1)
  (4): nn.SpatialMaxPooling(2,2,2,2)
  (5): nn.ReLU
  (6): nn.SpatialConvolutionMM(256 -> 384, 3x3, 1,1)
  (7): nn.SpatialMaxPooling(2,2,2,2)
  (8): nn.ReLU
  (9): nn.SpatialConvolutionMM(384 -> 384, 3x3, 1,1)
  (10): nn.SpatialMaxPooling(2,2,2,2)
  (11): nn.ReLU
  (12): nn.SpatialConvolutionMM(384 -> 256, 3x3, 1,1)
  (13): nn.ReLU
  (14): nn.View
  (15): nn.Dropout(0.500000)
  (16): nn.Linear(2304 -> 4096)
  (17): nn.ReLU
  (18): nn.Dropout(0.500000)
  (19): nn.Linear(4096 -> 4096)
  (20): nn.ReLU
  (21): nn.Dropout(0.500000)
  (22): nn.Linear(4096 -> 1000)
  (23): nn.SoftMax
}
   ```

### On visualization of TeraDeep 1000 Categories Net

Here we visualize the first layer filters (96 9x9 filters) and feature maps from first convolution layer
   
|                                               |
|:---------------------------------------------:|
|Filters from first layer                       |
|![filters](/data/first-layer-filters.png)      |
|Feature maps from first convolution layer      |
|![fms](/data/first-layer-feature-maps-lena.png)|
   
### Network structure after modifying TeraDeep 1000 Categories Net

```
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> output]
  (1): nn.SpatialConvolutionMM(3 -> 96, 9x9, 4,4, 4,4)
  (2): nn.SpikeReLU
  (3): nn.SpatialConvolutionMM(96 -> 256, 5x5)
  (4): nn.SpatialMaxPooling(2,2,2,2)
  (5): nn.SpikeReLU
  (6): nn.SpatialConvolutionMM(256 -> 384, 3x3)
  (7): nn.SpatialMaxPooling(2,2,2,2)
  (8): nn.SpikeReLU
  (9): nn.SpatialConvolutionMM(384 -> 384, 3x3)
  (10): nn.SpatialMaxPooling(2,2,2,2)
  (11): nn.SpikeReLU
  (12): nn.SpatialConvolutionMM(384 -> 256, 3x3)
  (13): nn.SpikeReLU
  (14): nn.View
  (15): nn.Dropout(0.500000)
  (16): nn.Linear(2304 -> 4096)
  (17): nn.SpikeReLU
  (18): nn.Dropout(0.500000)
  (19): nn.Linear(4096 -> 4096)
  (20): nn.SpikeReLU
  (21): nn.Dropout(0.500000)
  (22): nn.Linear(4096 -> 1000)
  (23): nn.SpikeReLU
}
```
   
### On writing `SpikeReLU` class

+ I'm basically trying to replicate Dianel's sensor fusion work from [here](https://github.com/dannyneil/sensor_fusion_iscas_2016) right now.
+ From the original code, Danny implemented spiking layer for dense layer, convolution layer and pooling layer. It's constrained by the model and the software he chose.
+ Instead of re-implementing these layers in Torch entirely, I figured I can write a general spiking ReLU class and replace the original ReLU functions
+ But, there are few problems, the first is on the time sync. In Theano, the time tensor can be traveled through computation graph. However, this is not in the case of Torch, so I defined a internal clock to every `SpikeReLU` object and update them at the same time.
+ Second is about pooling method, previous papers told me using Max-pooling is rather difficult to implement (or messy). But here I haven't met the tricky part yet.
+ The testing part is rather tricky, I would like to give a single image first and see if there is reasonable output. Hook this architecture with camera is rather a later work.  
+ Right now, each `SpikeReLU` object is initialized with manual setup. There should be a way to dress up the network without this kind of intervention.
+ For the bigger picture, not all ConvNets are for Object recognition, and not all ConvNets are only having simple convolution and pooling method. Dealing with that should be interesting.
+ This technique doesn't really reduce the amount of data and space used. I would be interested to see how this can be a real energy saving plan (notice all data flow is still in float numbers) 
+ Setting this modified network up with a huge dataset like ImageNet should be interesting.

### On methodology and general concerns

+ Danny's code goes `ConvLayer-->ReLU-->pooling Layer` in this design. However, the typical design is `ConvLayer-->pooling Layer-->ReLU`. They are doing the same job in conventional ConvNets. Following the later one could result simpler implementation (which in this repo), pooling layer is just one subroutine of getting output from a convolution layer.

### On plotting feature maps from Spiking ConvNet

Features maps of second ConvLayer after pooling. Each feature map has a size of 26x26, and in total 256 feature maps. The input image is the standard Lena image resized to 224x224. Following table prints feature maps in 20 time steps (the configuration follows Danny's code from [here](https://github.com/dannyneil/sensor_fusion_iscas_2016/blob/master/test_convnet.py#L36-L41):

**These pictures are generated with original Lena image without taking spike snapshot**

|                                             |                                             |                                             |
|:-------------------------------------------:|:-------------------------------------------:|:-------------------------------------------:|
|1                                            |2                                            |3                                            |
|![ts 1](/data/spike-conv-layer-5-out-1.png)  |![ts 2](/data/spike-conv-layer-5-out-2.png)  |![ts 3](/data/spike-conv-layer-5-out-3.png)  |
|4                                            |5                                            |6                                            |
|![ts 4](/data/spike-conv-layer-5-out-4.png)  |![ts 5](/data/spike-conv-layer-5-out-5.png)  |![ts 6](/data/spike-conv-layer-5-out-6.png)  |
|7                                            |8                                            |9                                            |
|![ts 7](/data/spike-conv-layer-5-out-7.png)  |![ts 8](/data/spike-conv-layer-5-out-8.png)  |![ts 9](/data/spike-conv-layer-5-out-9.png)  |
|10                                           |11                                           |12                                           |
|![ts 10](/data/spike-conv-layer-5-out-10.png)|![ts 11](/data/spike-conv-layer-5-out-11.png)|![ts 12](/data/spike-conv-layer-5-out-12.png)|
|13                                           |14                                           |15                                           |
|![ts 13](/data/spike-conv-layer-5-out-13.png)|![ts 14](/data/spike-conv-layer-5-out-14.png)|![ts 15](/data/spike-conv-layer-5-out-15.png)|
|16                                           |17                                           |18                                           |
|![ts 16](/data/spike-conv-layer-5-out-16.png)|![ts 17](/data/spike-conv-layer-5-out-17.png)|![ts 18](/data/spike-conv-layer-5-out-18.png)|
|19                                           |20                                           |                                             |
|![ts 19](/data/spike-conv-layer-5-out-19.png)|![ts 20](/data/spike-conv-layer-5-out-20.png)|                                             |
   
The values in these features maps appear to be binary. And as the trail goes, the activations (1s) in feature maps tend to appear less.

### On first attempt of replicate [Danny's experiment](https://github.com/dannyneil/sensor_fusion_iscas_2016/blob/master/spike_tester_theano.py) on single image 

The Lena image is processed with `spike_snapshot`, and results in this presentation

|                                                           |
|:---------------------------------------------------------:|
|![Lena spike snapshot](/data/lena-after-spike-snapshot.png)|

While I ran the experiments, I checked two things:

+ Since I don't have a recursive reset as original code, I did a manual reset for `SpikeReLU`, and works.
+ I also don't have a global time clock that can be referred by previous layer, I keep a internal clock at every `SpikeReLU`, and it's synced.

I took 20 rounds on the same image, and for each round, there are 20 steps like previous setup. All parameters are followed. Here I showed feature maps of second ConvLayer after pooling from first and second round.
Although the time is flying, feature maps of same time step in different rounds seem having no difference.

|   |                                                     |                                                     |                                                     |
|:-:|:---------------------------------------------------:|:---------------------------------------------------:|:---------------------------------------------------:|
|   |1                                                    |2                                                    |3                                                    |
|1st|![ts 1](/data/exp_2/spike-conv-layer-5-out-1-1.png)  |![ts 2](/data/exp_2/spike-conv-layer-5-out-1-2.png)  |![ts 3](/data/exp_2/spike-conv-layer-5-out-1-3.png)  |
|2nd|![ts 1](/data/exp_2/spike-conv-layer-5-out-2-1.png)  |![ts 2](/data/exp_2/spike-conv-layer-5-out-2-2.png)  |![ts 3](/data/exp_2/spike-conv-layer-5-out-2-3.png)  |
|   |4                                                    |5                                                    |6                                                    |
|1st|![ts 4](/data/exp_2/spike-conv-layer-5-out-1-4.png)  |![ts 5](/data/exp_2/spike-conv-layer-5-out-1-5.png)  |![ts 6](/data/exp_2/spike-conv-layer-5-out-1-6.png)  |
|2nd|![ts 4](/data/exp_2/spike-conv-layer-5-out-2-4.png)  |![ts 5](/data/exp_2/spike-conv-layer-5-out-2-5.png)  |![ts 6](/data/exp_2/spike-conv-layer-5-out-2-6.png)  |
|   |7                                                    |8                                                    |9                                                    |
|1st|![ts 7](/data/exp_2/spike-conv-layer-5-out-1-7.png)  |![ts 8](/data/exp_2/spike-conv-layer-5-out-1-8.png)  |![ts 9](/data/exp_2/spike-conv-layer-5-out-1-9.png)  |
|2nd|![ts 7](/data/exp_2/spike-conv-layer-5-out-2-7.png)  |![ts 8](/data/exp_2/spike-conv-layer-5-out-2-8.png)  |![ts 9](/data/exp_2/spike-conv-layer-5-out-2-9.png)  |
|   |10                                                   |11                                                   |12                                                   |
|1st|![ts 10](/data/exp_2/spike-conv-layer-5-out-1-10.png)|![ts 11](/data/exp_2/spike-conv-layer-5-out-1-11.png)|![ts 12](/data/exp_2/spike-conv-layer-5-out-1-12.png)|
|2nd|![ts 10](/data/exp_2/spike-conv-layer-5-out-2-10.png)|![ts 11](/data/exp_2/spike-conv-layer-5-out-2-11.png)|![ts 12](/data/exp_2/spike-conv-layer-5-out-2-12.png)|
|   |13                                                   |14                                                   |15                                                   |
|1st|![ts 13](/data/exp_2/spike-conv-layer-5-out-1-13.png)|![ts 14](/data/exp_2/spike-conv-layer-5-out-1-14.png)|![ts 15](/data/exp_2/spike-conv-layer-5-out-1-15.png)|
|2nd|![ts 13](/data/exp_2/spike-conv-layer-5-out-2-13.png)|![ts 14](/data/exp_2/spike-conv-layer-5-out-2-14.png)|![ts 15](/data/exp_2/spike-conv-layer-5-out-2-15.png)|
|   |16                                                   |17                                                   |18                                                   |
|1st|![ts 16](/data/exp_2/spike-conv-layer-5-out-1-16.png)|![ts 17](/data/exp_2/spike-conv-layer-5-out-1-17.png)|![ts 18](/data/exp_2/spike-conv-layer-5-out-1-18.png)|
|2nd|![ts 16](/data/exp_2/spike-conv-layer-5-out-2-16.png)|![ts 17](/data/exp_2/spike-conv-layer-5-out-2-17.png)|![ts 18](/data/exp_2/spike-conv-layer-5-out-2-18.png)|
|   |19                                                   |20                                                   |                                                     |
|1st|![ts 19](/data/exp_2/spike-conv-layer-5-out-1-19.png)|![ts 20](/data/exp_2/spike-conv-layer-5-out-1-20.png)|                                                     |
|2nd|![ts 19](/data/exp_2/spike-conv-layer-5-out-2-19.png)|![ts 20](/data/exp_2/spike-conv-layer-5-out-2-20.png)|                                                     |

The output log contains number of rounds, internal clock timing and labels predicted:
```
-------------------------	1
Internal time clock: 	0.021	s
ad
childs
design
drive
id
medium
toy
-------------------------

...............
...............
...............

-------------------------	19
Internal time clock: 	0.381	s
ad
childs
design
drive
id
medium
toy
-------------------------

-------------------------	20
Internal time clock: 	0.401	s
ad
childs
design
drive
id
medium
toy
-------------------------
```

The result does not seem reasonable enough, especially there is a female inside the picture, but it's still partially making sense.
Furthermore, since the output from last `SpikeReLU` layer is only spikes, so there is no probability distribution as provided by softmax function.
So I couldn't rank top guesses. So in this case, I printed all possible labels that are indicated by output vector.

And the output labels seems not changing over time, I don't know if it's caused by this still image setting (maybe should introduce some random shifts).

### On MNIST classification with `SpikeReLU` class

I was going to test CIFAR-10 or other datasets instead of MNIST initially. However there is not much pre-trained Caffe model or Torch model available.
One Caffe model for CIFAR-10 is from the paper _Network in Network_, I loaded and looked up the structure, it's kind of complex for a starter.

Finally I landed with Caffe's example network --- LeNet. And the structure of this network is similar to Danny's proposal

```
CONV      20 5x5  --> 20 24x24
MaxPool      2x2  --> 20 12x12
CONV      50 5x5  --> 50  8x8
MaxPool      2x2  --> 50  4x4
Flatten
Linear       500  --> 1   500
ReLU
Linear       10   --> 1   10
Softmax
``` 

This network is reporting 99.14% accuracy originally. However, I noticed that there is no activation function after 2 ConvLayers nor pooling layers.
Firstly, I figured I should add `SpikeReLU` after every pooling layer, and then replace `ReLU` and `Softmax` function as `SpikeReLU`. However, this did not go well.

Then I thought maybe I shouldn't plug `SpikeReLU` after pooling layer, for the matter of fact, it changes how this network is trained initially. The first four may just serve as a feature extractor.
So I changed the `ReLU` and `Softmax` function only. This didn't go anywhere good either.

At this point, I remembered something important I've not done --- normalization on weights.
This globel model based normalization seems the trick getting higher accuracies. But well, after I applied normalization (the factor is 4000+), all weights are very small.
Well, in a way, this does not harm your convolution, but it seems making spikes impossible to fire (pure black feature maps are plotted through the time)
I thought, maybe my bias is hurting me. Because apparently, large bias and small weights are not working together.

Right now, I'm out of my moves. I need to exam again the situation and run some more tests to figure out what's really inside there:

+ Run original network without bias
+ Try to normalize bias as weights
+ Checking the code of `SpikeReLU`
+ Checking the code of recognition part
+ You can remove bias and change max-pooling to average-pooling to trained ConvNets without a problem?

One interesting discovery was although the prediction based on maximum spikes is not working well,
the right prediction always spikes somewhere in the epochs. If it's a random behavior, then it should give me lower accuracy, but the correction prediction always spikes.

There are some mistakes I found while I'm trying with this network:

+ One big mistake is I ignored the fact that variable and memory management is different between Torch and Python. I wrote straight away like Python, but it's kind of a wrong idea.
  The problem is when Torch does variable assignment like `A=B` on tensors, they automatically connected, so whatever I change something on `B`, `A` would go exactly the same way.
  
+ The second problem is I should predicted the result for every epochs, instead, I should accumulate the results from each epochs and then do prediction for once

+ The third problem is the spike snapshot should be changed for every epochs, instead, I used a static one before.  

---
---

**Start Over and Check***
 
### On Spike Generation

The images are preprocessed by taking _spike snapshots_ before fed into Spiking NNs.
This imitates behavior of event vision sensor (but far from real). Two examples are presented here:

1. The first one is taken from test dataset of MNIST
2. The second one is a rescaled version of Lena image

Both images are generated based on following rule:

```
rand(0,1)<0.33*Image
```

|                                                   |
|:-------------------------------------------------:|
|MNIST Example for digit 7                          |
|![mnist example](/data/spike-gen-mnist-exp-7.gif)  |
|Lena Example for color image                       |
|![lena example](/data/spike-gen-lena-color.gif)    |

### Tailor LeNet

Current papers on Spiking ConvNets are using average pooling in order to simplify the process.
So before I figured out how to perform max-pooling, I just simply turned max-pooling in LeNet to average-pooling.

According to the paper "Spiking Deep Convolutional Neural Networks for Energy-Efficient Object Recognition", the major difficulty is:

> Max-pooling requires two layers of spiking networks. In CNN, spatial max-pooling is implemented as taking the 
> maximum output values over a small image neighborhood in the imput. In SNN, we need two-layer neural networks to
> accomplish this, with lateral inhibition followed by pooling over these small image regions. This approach
> requires more neurons and can cause accuracy loss due to the added complexity.

The LeNet I'm using reports 99.14% originally. Then after I shutdown bias, the performance drops to 98.68%.
I then replaced max-pooling to average-pooling, the performance now is 69.8%.
Of course, this dramatic performance drop is expected. And I use this as a baseline to test
my `SpikeReLU` correctness.

### On MNIST Classification using tailored LeNet (Second Attempt)

In this experiment, instead of using Danny's code, I appiled algorithms from the paper 
"Spiking Deep Convolutional Neural Networks for Energy-Efficient Object Recognition".
The implementation is simpler than Danny's. In particular, it does not use any time factor.

After I tailored the network as suggested by the paper, I then setup one experiment. And the result is quite surprising: 63.8%
This accuracy is similar to the original network. Furthermore, all spiking threshold is set as 1, and there is no further tuning included.

Here is the confusion matrix generated:

```
ConfusionMatrix:
[[     956       0       0       2       0       7       0       9       6       0]   97.551% 
 [       0     557       0     265       0       0       0       0     312       1]   49.075% 
 [      24      34     463      76      40       8      34      74     279       0]   44.864% 
 [       9       9       1     805      12       7       0      12     132      23]   79.703% 
 [       3       6       5       4     889       0       5      41      16      13]   90.530% 
 [      39      17       0      58      55     373       2      21     308      19]   41.816% 
 [      55      25      11      21      90     140     500      34      82       0]   52.192% 
 [       4       6       2      38      66       0       0     855      43      14]   83.171% 
 [       6       2       0      59      13       4       1      11     878       0]   90.144% 
 [       9       6       0      18     611       3       1     179      78     104]]  10.307% 
 + average row correct: 63.935313522816% 
 + average rowUcol correct (VOC measure): 46.791453659534% 
 + global correct: 63.8%
```

 
## Contacts

Hu Yuhuang  
Email: duguyue100@gmail.com
