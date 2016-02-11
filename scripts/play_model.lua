--[[
Author: Yuhuang Hu
Email : duguyue100@gmail.com

Notes : this script is testing a pretrained network using nn package.
]]

package.path = package.path .. ";../?.lua"

require 'nn';
require 'SpikeReLU';
require 'image';
require 'csvigo';

function topk(data, k)

  local data_t=data;
 
  print (data_t:size()) 
  idx=torch.zeros(k,2);
  
  for i=1,k do
    m, id = data_t:max(1);
    idx[i][1]=id[1];
    idx[i][2]=m[1]
    data_t[id[1]]=-1;
  end
  
  return idx;
end

--- find all 1s in result data
function findall(data)
  
  idx=torch.zeros(1000);
  j=1;
  size=data:size();
  print (size)
  for i=1,1000 do
    if data[i]==1 then
      idx[j]=i;
      j=j+1;
    end
  end
  
  return idx;

end

--[[ Load Network  ]]
local network = {};

network.model=torch.load('../data/model.net');
network.stat=torch.load('../data/stat.t7');
network.label=csvigo.load({path='../data/categories.txt', mode="large"});
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
--l=image.load("../data/apple.jpg");
--l=image.load("../data/lego_robot_1.jpg");
l=image.scale(l, 224, 224);

num_batches=20;
dt=0.001;
max_rate=1000.0;
rescale_fac = 1./(dt * max_rate);

spike_snapshot=torch.randn(3,224,224)*rescale_fac;

l=spike_snapshot:le(l):float();

for k=1, num_batches do
  
  --- cleaning SpikeReLU
  
  network.model.modules[2]:reset()
  network.model.modules[5]:reset()
  network.model.modules[8]:reset()
  network.model.modules[11]:reset()
  network.model.modules[13]:reset()
  network.model.modules[17]:reset()
  network.model.modules[20]:reset()
  network.model.modules[23]:reset()
   
  for i=1,20 do  
    network.model:forward(l:float());
    image_title=string.format("../data/exp_2/spike-conv-layer-5-out-%i-%i.png", k, i);
    print (image_title)
    fms=network.model.modules[5].output; -- call the layer output you want.
    image.save(image_title, image.toDisplayTensor{input=fms, padding=1, nrow=16, scaleeach=true});
  end
  
  out=network.model.modules[23].output;
  all_labels=findall(out);
  print ("-------------------------", k)
  print ("Internal time clock: ", network.model.modules[2].time, "s")
  for i=1, 1000 do
    if all_labels[i]~=0 then
      print (network.label[all_labels[i]+1][1]);
    end
  end
  print ("-------------------------")
end

--out=network.model.modules[23].output;
--
--all_labels=findall(out);
--
--for i=1, 1000 do
--  if all_labels[i]~=0 then
--    print (network.label[all_labels[i]+1][1]);
--  end
--end

--print (network.label[2][1])

--fms=network.model.modules[5].output; -- call the layer output you want.
--image.display(image.toDisplayTensor{input=fms, padding=1, nrow=16, scaleeach=true});

--[[ Print model ]]

--print (network.model);