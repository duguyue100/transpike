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
+ Figure out a way of hooking this modified network with a camera [TODO]
+ Setup experiments as indicated in Python code [TODO]

## Notes

+ Network Structure of TeraDeep's 1000 Categories Net

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

+ On writing `SpikeReLU` class

   + I'm basically trying to replicate Dianel's sensor fusion work from [here](https://github.com/dannyneil/sensor_fusion_iscas_2016) right now.
   + From the original code, Danny implemented spiking layer for dense layer, convolution layer and polling layer. It's constrained by the model and the software he chose.
   + Instead of re-implementing these layers in Torch entirely, I figured I can write a general spiking ReLU class and replace the original ReLU functions
   + But, there are few problems, the first is on the time sync. In Theano, the time tensor can be traveled through computation graph. However, this is not in the case of Torch, so I defined a internal clock to every `SpikeReLU` object and update them at the same time.
   + Second is about polling method, previous papers told me using Max-Polling is rather difficult to implement (or messy). But here I haven't met the tricky part yet.
   + The testing part is rather tricky, I would like to give a single image first and see if there is reasonable output. Hook this architecture with camera is rather a later work.  
   + Right now, each `SpikeReLU` object is initialized with manual setup. There should be a way to dress up the network without this kind of intervention.
   + For the bigger picture, not all ConvNets are for Object recognition, and not all ConvNets are only having simple convolution and polling method. Dealing with that should be interesting.
   + This technique doesn't really reduce the amount of data and space used. I would be interested to see how this can be a real energy saving plan (notice all data flow is still in float numbers) 
   + Setting this modified network up with a huge dataset like ImageNet should be interesting.
   
+ On methodology and general concerns

   + Danny's code put ReLU after every convolution layers and polling layer. However the typical way of doing so is somehow glue convolution layer and polling layer as one, and it behaves exactly the same. So in this sense, polling should be as a subroutine to provide output of a layer instead of a particular spiking layer.

## Contacts

Hu Yuhuang  
Email: duguyue100@gmail.com
