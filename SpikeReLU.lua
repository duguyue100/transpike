require 'nn'

local SpikeReLU, Parent = torch.class('nn.SpikeReLU', 'nn.Module')

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
-- timestep : float
--    internal time step, however entire structure shares same clock.
-- 
function SpikeReLU:__init(pre_size, timestep, threshold, refractory)
  Parent.__init(self);
  self.pre_size=pre_size;
  self.threshold=threshold or 1.0;
  self.refractory=refractory or 1.0;
  self.timestep=timestep or 0.001;
  self.time=self.timestep;
  
  self.mem=torch.zeros(self.pre_size):float();
  self.refrac_until=torch.zeros(self.pre_size):float();
end

--- This function resets self.mem and self.refrac_until variables
-- A systematic search is needed in order to realize original codes
-- recursive reset, find here:
-- https://github.com/dannyneil/sensor_fusion_iscas_2016/blob/master/theano_layers.py#L121
-- https://github.com/dannyneil/sensor_fusion_iscas_2016/blob/master/spike_tester_theano.py#L13
function SpikeReLU:reset()
  self.mem:zero():float();
--  self.refrac_until:zero():float();
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
function SpikeReLU:updateOutput(input)
  self.output=input;
  
  mem_new=self.mem+input;
  mem_new[mem_new:le(0.)]=0.0;
  
  output_spikes=mem_new:clone();
  output_spikes[output_spikes:ge(self.threshold)]:float();
  
  mem_new[mem_new:ge(self.threshold)]=0.0;
  self.mem=mem_new:clone();
  
  self.output=output_spikes;

--  -- Destroy impulse if in refrac
--  masked_imp=input:clone();
--  masked_imp[self.refrac_until:gt(self.time)]=0.0;
--  
--  -- Add impulse
--  new_mem=self.mem+masked_imp:float();
--  
--  -- Store spiking
--  output_spikes=new_mem:clone():gt(self.threshold):float();
--  
--  -- Reset neuron
--  new_and_reset_mem=new_mem:clone();
--  new_and_reset_mem[output_spikes:gt(0)]=0.0;
--  
--  -- Store refractory
--  new_refractory=self.refrac_until:clone();
--  new_refractory[output_spikes:gt(0)]=self.time+self.refractory;
--
--  -- Updates 
--  self.refrac_until=new_refractory:clone();
--  self.mem=new_and_reset_mem:clone();
--  
--  -- Renew system time
--  self.time=self.time+self.timestep;
--  
--  self.output=output_spikes;
  
  return self.output;
end

--- Since it's only transferring from ReLU out to spikes
-- therefore gradInput is not changed as gradOutput
function SpikeReLU:updateGradInput(input, gradOutput)
  self.gradInput=gradOutput;
  return self.gradInput;
end

------- Setting and Clearing functions ------

--- Set output size of previous layer.
-- doc found at init function
function SpikeReLU:setPreSize(pre_size)
  self.pre_size=pre_size;
end

--- Set threshold variable
-- doc found at init function
function SpikeReLU:setThreshold(threshold)
  self.threshold=threshold;
end

--- Set refractory variable
-- doc found at init function
function SpikeReLU:setRefractory(refractory)
  self.refractory=refractory;
end

--- Set timestep variable
-- doc found at init function
function SpikeReLU:setTimeStep(timestep)
  self.timestep=timestep
end

function SpikeReLU:clearState()
  return Parent.clearState(self);
end