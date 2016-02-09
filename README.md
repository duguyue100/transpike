# TranSpike

This is my first attempt to build a transformer for Spiking Neural Networks.
The idea is roughly following the paper *Fast-Classifying, High-Accuracy Spiking Deep Networks Through Weight and Threshold Balancing*.

This project is purely built by using Torch 7.

## Updates

+ Load pretrained VGG-16 network or TeraDeep Network [2016-02-09]
+ Print VGG-16 model, filters, and input, output stream [2016-02-09]
+ Tackle matlab code for transfering ConvNet to SNNs [TODO]

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

## Contacts

Hu Yuhuang  
Email: duguyue100@gmail.com
