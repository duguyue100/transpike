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

--- Model based Normalization

function sum_module(module)
  weight=module.weight:clone();
  weight[weight:le(0)]=0;
  
  return torch.sum(weight);
end

------ Load Model and Prepare Data ------
prototxt='../data/lenet.prototxt'
binary='../data/lenet_iter_10000.caffemodel'
net = loadcaffe.load(prototxt, binary)
testData = mnist.testdataset()
images = testData.data:float():div(256)
net:evaluate();
net:float()

----- Normalize Network ------

max_weight_sum=math.max(sum_module(net.modules[1]), sum_module(net.modules[3]), sum_module(net.modules[6]), sum_module(net.modules[8]));

--net.modules[1].weight=net.modules[1].weight/max_weight_sum;
--net.modules[3].weight=net.modules[3].weight/max_weight_sum;
--net.modules[6].weight=net.modules[6].weight/max_weight_sum;
--net.modules[8].weight=net.modules[8].weight/max_weight_sum;

--net.modules[1].bias=net.modules[1].bias:zero();
--net.modules[3].bias=net.modules[3].bias:zero();
--net.modules[6].bias=net.modules[6].bias:zero();
--net.modules[8].bias=net.modules[8].bias:zero();

-- print (net.modules[1].weight)

------ Modify Networks ------
--
-- CONV      20 5x5  --> 20 24x24
-- MaxPool      2x2  --> 20 12x12
-- SpikeReLU
-- CONV      50 5x5  --> 50  8x8
-- MaxPool      2x2  --> 50  4x4
-- SpikeReLU
-- Flatten
-- Linear       500  --> 1   500
-- SpikeReLU
-- Linear       10   --> 1   10
-- SpikeReLU

net:insert(nn.SpikeReLU(torch.LongStorage{20, 12, 12}), 3);
net:insert(nn.SpikeReLU(torch.LongStorage{50, 4, 4}), 6);
net:remove(9);
net:insert(nn.SpikeReLU(torch.LongStorage{500}), 9);
net:insert(nn.SpikeReLU(torch.LongStorage{10}), 11);

------ Set Up Experiment ------

num_images=images:size();
dt=0.001;
max_rate=1000.0;
rescale_fac = 1./(dt * max_rate);
output_spike=torch.zeros(10):int();

score=0.0;
for k=1,1 do
  
  ------ Reset Network ------
  net.modules[3]:reset();
  net.modules[6]:reset(); 
  net.modules[9]:reset();
  net.modules[11]:reset();
  
  output_spike:zero();
  
  for i=1,1 do
    l=images[k]:view(1,28,28);
--    spike_snapshot=torch.randn(1,28,28)*rescale_fac;
--    l=spike_snapshot:le(images[k]:view(1,28,28):double()):float();
    net:forward(l);
    
    output_spike=output_spike+net.modules[11].output:int();
  end
  
  fms=net.modules[6].output;
  image.display(image.toDisplayTensor{input=fms, padding=1, nrow=5, scaleeach=true});
    
    
  _, idx=output_spike:max(1);
  print ("---------------------------------------", k)
  print ("Prediction: ", idx[1]-1)
  print ("Actual    : ", testData.label[k])
  print ("---------------------------------------")
  
--  if (output_spike[testData.label[k]+1]) then
--    score=score+1;
--  end
  
  if (idx[1]-1)==testData.label[k] then
    score=score+1
  end;
  
end

print (score/200.)
--image.display(image.toDisplayTensor{input=fms, padding=1, nrow=5, scaleeach=true});

-- will be used to print the results
--confusion = optim.ConfusionMatrix(10)
--
--for i=1,images:size(1) do
--  _,y = net:forward(images[i]:view(1,28,28)):max(1)
--  confusion:add(y[1], testData.label[i]+1)
--end
--
---- that's all! will print the error and confusion matrix
--print(confusion)