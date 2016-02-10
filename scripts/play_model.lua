--[[
Author: Yuhuang Hu
Email : duguyue100@gmail.com

Notes : this script is testing a pretrained network using nn package.
]]

package.path = package.path .. ";../?.lua"

require 'nn';
require 'SpikeReLU';
require 'image';

--[[ Load Network  ]]
local network = {};

network.model=torch.load('../data/model.net');
network.stat=torch.load('../data/stat.t7');
network.model:evaluate();

-- remove ReLU and plug SpikeReLU
network.model:remove(2);
network.model:insert(nn.SpikeReLU(torch.LongStorage{96, 56, 56}), 2);
network.model:remove(5);
network.model:insert(nn.SpikeReLU(torch.LongStorage{256, 26, 26}), 5);
network.model:remove(8);
network.model:insert(nn.SpikeReLU(torch.LongStorage{384, 12, 12}), 8);
network.model:remove(11);
network.model:insert(nn.SpikeReLU(torch.LongStorage{384, 5, 5}), 11);
network.model:remove(13);
network.model:insert(nn.SpikeReLU(torch.LongStorage{256, 3, 3}), 13);
network.model:remove(17);
network.model:insert(nn.SpikeReLU(torch.LongStorage{4096}), 17);
network.model:remove(20);
network.model:insert(nn.SpikeReLU(torch.LongStorage{4096}), 20);

-- modify last layer
network.model:remove(23);
network.model:insert(nn.SpikeReLU(torch.LongStorage{1000}), 23);

--torch.save("../data/model_spike.net", network.model);
--print ("save completed");

l=image.lena();
l=image.scale(l, 224, 224);

for i=1,20 do
  network.model:forward(l:float());
end

fms=network.model.modules[5].output; -- call the layer output you want.
image.display(image.toDisplayTensor{input=fms, padding=1, nrow=16, scaleeach=true});

--network.model:insert(nn.ReLU(), 23);
--network.model:insert(nn.Dropout(0.500000), 24);
--network.model:insert(nn.Linear(1000, 1000), 25);

--[[ Print model ]]

print (network.model);
--maxs, indices = torch.max(network.model.modules[23].output)
--
--print (maxs);
--print (indices);

--print (network.model.modules[23].output);