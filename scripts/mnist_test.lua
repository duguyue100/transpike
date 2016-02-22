--[[
Author: Yuhuang Hu
Email : duguyue100@gmail.com

Notes : this script took LeNet of MNIST as example of testing recognition
]]

package.path = package.path .. ";../?.lua"


require 'xlua'
require 'optim'
require 'nn'
require 'loadcaffe'
require 'image'
require 'SpikeReLU'
mnist = require 'mnist'

------ Load Model and Prepare Data ------
prototxt='../data/lenet.prototxt'
binary='../data/lenet_iter_10000.caffemodel'
net = loadcaffe.load(prototxt, binary)
testData = mnist.testdataset()
images = testData.data:float():div(256)
net:evaluate();
net:float()

------ Tailor ConvNets ------

-- Shutdown bias
net.modules[1].bias:zero();
net.modules[3].bias:zero();
net.modules[6].bias:zero();
net.modules[8].bias:zero();

-- replace to average-pooling
net:remove(2);
net:insert(nn.SpatialAveragePooling(2,2,2,2,0,0):ceil(), 2);
net:remove(4);
net:insert(nn.SpatialAveragePooling(2,2,2,2,0,0):ceil(), 4);

------ Modify Networks ------
--
-- CONV        20 5x5  --> 20 24x24
-- AveragePool    2x2  --> 20 12x12
-- SpikeReLU
-- CONV        50 5x5  --> 50  8x8
-- AveragePool    2x2  --> 50  4x4
-- SpikeReLU
-- Flatten
-- Linear         500  --> 1   500
-- SpikeReLU
-- Linear         10   --> 1   10
-- SpikeReLU

--net:insert(nn.SpikeReLU(torch.LongStorage{20, 12, 12}), 3);
--net:insert(nn.SpikeReLU(torch.LongStorage{50, 4, 4}), 6);
--net:remove(9);
--net:insert(nn.SpikeReLU(torch.LongStorage{500}), 9);
--net:insert(nn.SpikeReLU(torch.LongStorage{10}), 11);

net:remove(7);
net:insert(nn.SpikeReLU(torch.LongStorage{500}), 7);
net:insert(nn.SpikeReLU(torch.LongStorage{10}), 9);

net:float();

------ Set Up Experiment ------

num_images=images:size();
dt=0.001;
max_rate=1000.0;
rescale_fac = 1./(dt * max_rate);
output_spike=torch.zeros(10):int();

score=0.0;

confusion = optim.ConfusionMatrix(10)

for k=1,num_images[1] do
  
  ------ Reset Network ------
  net.modules[7]:reset();
  net.modules[9]:reset();
  
  output_spike:zero();
  
  
  for i=1,20 do
    l=images[k]:view(1,28,28):float();
    spike_snapshot=torch.rand(1,28,28):float();
    l=spike_snapshot:le(l*0.33):float();
    net:forward(l);
  
    output_spike=output_spike+net.modules[9].output:int();
  end
  
--  fms=net.modules[6].output;
--  image.display(image.toDisplayTensor{input=fms, padding=1, nrow=5, scaleeach=true});

  _, idx=output_spike:max(1);
  print ("---------------------------------------", k)
  print ("Prediction: ", idx[1]-1)
  print ("Actual    : ", testData.label[k])
  print ("---------------------------------------")
  
  confusion:add(idx[1], testData.label[k]+1)
    
  if (idx[1]-1)==testData.label[k] then
    score=score+1
  end;
end

print(confusion)

print (score);
print (score/num_images[1])