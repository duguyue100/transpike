--[[
Author: Yuhuang Hu
Email : duguyue100@gmail.com

Notes : this script is testing a pretrained network using nn package.
]]

require 'nn';
require 'image';

l=image.lena();
l=image.scale(l, 252, 240);
--image.display(l);

local network = {};

network.model=torch.load('../data/model.net');
network.stat=torch.load('../data/stat.t7');

out=network.model:forward(l:float())

--print (network.model.modules[1].output);
print (out);
--print (network.model);

--print (network.stat.std);