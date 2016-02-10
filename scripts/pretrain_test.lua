--[[
Author: Yuhuang Hu
Email : duguyue100@gmail.com

Notes : this script is testing a pretrained network using nn package.
]]

require 'nn';
require 'image';

--[[ Load Network  ]]
local network = {};

network.model=torch.load('../data/model.net');
network.stat=torch.load('../data/stat.t7');


--[[ visualize first ConvLayer weights]]

filters=network.model.modules[1].weight; -- call the weights you want
filters=torch.reshape(filters, torch.LongStorage{96,3,9,9});

print (network.model.modules[1].bias);

out=image.toDisplayTensor{input=filters, padding=1, nrow=12, scaleeach=true};
image.save("../data/first-layer-filters.png", out);

--[[ visualize first ConvLayer after polling feature maps]]

-- Lena image
l=image.lena();
l=image.scale(l, 224, 224);

network.model:forward(l:float());
fms=network.model.modules[2].output; -- call the layer output you want.
image.save("../data/first-layer-feature-maps-lena.png", image.toDisplayTensor{input=fms, padding=1, nrow=12, scaleeach=true});

-- Local image
f=image.load("../data/lego_robot_1.jpg");
f=image.scale(f, 224, 224);

-- image will change every time when different image is fed.
network.model:forward(f:float());
fms=network.model.modules[2].output; -- call the layer output you want.
image.save("../data/first-layer-feature-maps-robot.png", image.toDisplayTensor{input=fms, padding=1, nrow=12, scaleeach=true});

