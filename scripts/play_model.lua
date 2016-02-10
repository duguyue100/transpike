--[[
Author: Yuhuang Hu
Email : duguyue100@gmail.com

Notes : this script is testing a pretrained network using nn package.
]]

package.path = package.path .. ";../?.lua"

require 'nn';
require 'SpikeConvReLU';
require 'image';

--[[ Load Network  ]]
local network = {};

network.model=torch.load('../data/model.net');
network.stat=torch.load('../data/stat.t7');
network.model:evaluate();

network.model:insert(nn.SpikeConvReLU(torch.LongStorage{96, 56, 56}), 2);

l=image.lena();
l=image.scale(l, 224, 224);
network.model:forward(l:float());

fms=network.model.modules[3].output; -- call the layer output you want.
print (fms:size())
--image.display(image.toDisplayTensor{input=fms, padding=1, nrow=12, scaleeach=true});

--network.model:insert(nn.ReLU(), 23);
--network.model:insert(nn.Dropout(0.500000), 24);
--network.model:insert(nn.Linear(1000, 1000), 25);

--[[ Print model ]]

print (network.model);