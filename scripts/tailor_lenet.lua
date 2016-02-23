require 'loadcaffe'
require 'xlua'
require 'optim'
mnist = require 'mnist'

-- to train lenet network please follow the steps
-- provided in CAFFE_DIR/examples/mnist
-- This is a testing commit for atom
-- This is a testing commit for atom again :)
prototxt = '../data/lenet.prototxt'
binary = '../data/lenet_iter_10000.caffemodel'

-- this will load the network and print it's structure
net = loadcaffe.load(prototxt, binary)

-- Tailor LeNet

net.modules[1].bias:zero();
net.modules[3].bias:zero();
net.modules[6].bias:zero();
net.modules[8].bias:zero();

-- After removing all bias, accuracy is 98.68%

net:insert(nn.ReLU(), 3);
net:insert(nn.ReLU(), 6);
--net:insert(nn.SpatialCrossMapLRN(1), 2);
--net:remove(3);
--net:insert(nn.SpatialAveragePooling(2,2,2,2,0,0):ceil(), 3);
--net:insert(nn.SpatialCrossMapLRN(1), 5);
--net:remove(6);
--net:insert(nn.SpatialAveragePooling(2,2,2,2,0,0):ceil(), 6);

net:remove(2);
net:insert(nn.SpatialAveragePooling(2,2,2,2,0,0):ceil(), 2);
net:remove(5);
net:insert(nn.SpatialAveragePooling(2,2,2,2,0,0):ceil(), 5);

-- After changing pooling method directly, accuracy is 69.8%

-- load test data
testData = mnist.testdataset()

-- preprocess by dividing by 256
images = testData.data:float():div(256)
net:float();

-- will be used to print the results
confusion = optim.ConfusionMatrix(10)

for i=1,images:size(1) do
  _,y = net:forward(images[i]:view(1,28,28)):max(1)
  confusion:add(y[1], testData.label[i]+1)
end

-- that's all! will print the error and confusion matrix
print(confusion)
