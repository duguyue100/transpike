require 'loadcaffe'
require 'xlua'
require 'optim'
require 'image'
mnist = require 'mnist'

-- Generate Spike from MNIST
testData = mnist.testdataset()
images = testData.data:float():div(256)

I=images[1]:view(1,28,28)*0.33;

--for i=1,50 do
--  spike_snapshot=torch.rand(1,28,28):float(); 
--  temp_I=spike_snapshot:le(I):float();
--  image_title=string.format("../data/spike_generation/spike-gen-mnist-exp-%i.png", i);
--  print (image_title);
--  image.save(image_title, temp_I);
--end

-- Generate Spike from Lena

l=image.lena();
l=image.scale(l, 224, 224):float()*0.33;

for i=1,50 do
  spike_snapshot=torch.rand(3,224,224):float();
  temp_l=spike_snapshot:le(l):float();
  
  image_title=string.format("../data/spike_generation/spike-gen-lena-%i.png", i);
  print (image_title);
  image.save(image_title, temp_l);
end



