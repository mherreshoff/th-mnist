mnist = require 'mnist'
nn = require 'nn'

function flattenImage(img)
  return img:view(28*28):double() / 255
end

function flattenInputIndicatorOutput(mnist_dataset)
  dataset = {}
  function dataset:size() return mnist_dataset.size end
  for i = 1,mnist_dataset.size do
    local input = flattenImage(mnist_dataset.data[i])
    local output = torch.zeros(10)
    output[mnist_dataset.label[i]+1] = 1
    dataset[i] = {input, output}
  end
  return dataset
end

perceptron = nn.Linear(28*28, 10)
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(perceptron, criterion)
trainer.learningRate = 0.01
trainer:train(flattenInputIndicatorOutput(mnist.traindataset()))

false_predictions = 0
test_set = mnist.testdataset()
for i = 1,test_set.size do
  local output = perceptron:forward(flattenImage(test_set.data[i]))
  local max, prediction_vec = torch.max(output, 1)
  prediction = prediction_vec[1] - 1
--  print(string.format('i=%d, prediction=%d reality=%d\n', i, prediction, test_set.label[i]))
  if prediction ~= test_set.label[i] then
    false_predictions = false_predictions + 1
  end
end

print(string.format('Error rate = %f%%', 100*(false_predictions/test_set.size)))

