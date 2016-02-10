require 'nn'

local SpikeConvReLU, Parent = torch.class('nn.SpikeConvReLU', 'nn.Module')

--- The init function set basic status of the layer
-- 
-- Parameters
-- ----------
-- pre_size : torch.LongStorage
--    size of output of previous layer
--    
-- threshold : float
-- 
-- refractory : float
-- 
function SpikeConvReLU:__init(pre_size, threshold, refractory)
  Parent.__init(self);
  self.pre_size=pre_size;
  self.threshold=threshold or 1.0;
  self.refractory=refractory or 1.0;
  
  self.mem=torch.zeros(self.pre_size);
  self.refrac_until=torch.zeros(self.pre_size);
end

--- This function resets self.mem and self.refrac_until variables
-- A systematic search is needed in order to realize original codes
-- recursive reset, find here:
-- https://github.com/dannyneil/sensor_fusion_iscas_2016/blob/master/theano_layers.py#L121
-- https://github.com/dannyneil/sensor_fusion_iscas_2016/blob/master/spike_tester_theano.py#L13
function SpikeConvReLU:reset()
  -- reset refrac_until and mem
  -- the original code is only Spiking layers, hence it can reset recursively
  -- by calling its parent, in this case, we need to do a systematic search
  -- and reset these two variables
  
  self.mem:zero();
  self.refrac_until:zero();
end

--- Update output spike to next layer
-- 
-- Parameters
-- ----------
-- input : torch.Tensor
--    the output tensor from last layer
-- 
-- Returns
-- -------
-- self.output : torch.Tensor
--    the input tensor for next layer (also as the output spikes)
function SpikeConvReLU:updateOutput(input)
  self.output=input;
  
  impulse=input;
  
  
  return self.output;
end

--- Since it's only transferring from ReLU out to spikes
-- therefore gradInput is not changed as gradOutput
function SpikeConvReLU:updateGradInput(input, gradOutput)
  self.gradInput=gradOutput;
  return self.gradInput;
end

------- Setting and Clearing functions ------

--- Set output size of previous layer.
-- doc found at init function
function SpikeConvReLU:setPreSize(pre_size)
  self.pre_size=pre_size;
end

--- Set threshold variable
-- doc found at init function
function SpikeConvReLU:setThreshold(threshold)
  self.threshold=threshold;
end

--- Set refractory variable
-- doc found at init function
function SpikeConvReLU:setRefractory(refractory)
  self.refractory=refractory;
end

function SpikeConvReLU:clearState()
  return Parent.clearState(self);
end