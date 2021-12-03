#include "DataSet.hpp"

#include <libkann/utilities/random.hpp>
#include <libkann/ConvolutionalLayer.hpp>
#include <libkann/NeuralNetwork.hpp>

#include <random>
#include <chrono>

static constexpr double LEARNING_RATE = 0.05;

int main()
{
  std::mt19937 generator(random<std::mt19937::result_type>());

  ConvolutionalLayer convolutionLayer(120, 100, 3, 4, 1);
  convolutionLayer.randomize(generator);

  ActivationLayer activationLayer(convolutionLayer.outputSize());
  activationLayer.activationFunction(ActivationFunction(ActivationFunction::Type::SIGMOID));

  Eigen::VectorXd input           = Eigen::VectorXd::Random(convolutionLayer.inputSize());
  Eigen::VectorXd expectedOutput  = Eigen::VectorXd::Random(activationLayer.outputSize());
  for(;;)
  {
    auto intermediate1 = convolutionLayer.feedForward(input);
    auto output = activationLayer.feedForward(intermediate1);

    Eigen::VectorXd outputGradient = 2.0 * (output - expectedOutput);
    std::cout << "Cost:" << outputGradient.dot(outputGradient) << std::endl;

    auto intermediate2 = activationLayer.backPropagate(outputGradient);
    auto inputGradient = convolutionLayer.backPropagate(intermediate2);
    convolutionLayer.train(0.01);
  }

  NeuralNetwork nn({DataSet::INPUT_LAYER_SIZE, 30, 30, 30, DataSet::OUTPUT_LAYER_SIZE}, generator, ActivationFunction(ActivationFunction::Type::SIGMOID));

  DataSet testingDataSet(
    "examples/backpropagation/datasets/t10k-images-idx3-ubyte",
    "examples/backpropagation/datasets/t10k-labels-idx1-ubyte"
  );
  DataSet trainingDataSet(
    "examples/backpropagation/datasets/train-images-idx3-ubyte",
    "examples/backpropagation/datasets/train-labels-idx1-ubyte"
  );

  testingDataSet.test(nn);
  trainingDataSet.train(nn, LEARNING_RATE);
  testingDataSet.test(nn);
}
