#include <libkann/NeuralNetwork.hpp>

#include <cmath>
#include <random>
#include <chrono>

#include <iostream>

double testFunction(double i)
{
  return std::sin(i);
  //if(i>0.0)
  //  return 1.0;

  //if(i<0.0)
  //  return -1.0;

  //return 0.0;
}

double errorFunction(double expected, double real)
{
  auto difference = expected - real;
  return std::sqrt(difference * difference);
}

static constexpr size_t BATCH_SIZE = 5;
static constexpr double LEARNING_RATE = 0.1;

int main()
{
  std::mt19937 generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> inputDist(0.0, M_PI);

  double minimumError = std::numeric_limits<double>::infinity();

  NeuralNetwork nn({1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1}, generator);
  for(size_t i=0;;++i)
  {
    double totalError = 0.0;
    for(size_t i=0; i<BATCH_SIZE; ++i)
    {
      auto input = inputDist(generator);
      auto expectedOutput = testFunction(input);

      nn.input(static_cast<size_t>(0), input);
      nn.feedForward();

      // Log error
      auto output = nn.output(static_cast<size_t>(0));
      totalError += errorFunction(expectedOutput, output);

      Eigen::VectorXd expectedOutputVector(1);
      expectedOutputVector(0) = expectedOutput;
      nn.backPropagate(expectedOutputVector);
    }
    double error = totalError / BATCH_SIZE;
    if(error<minimumError)
    {
      minimumError = error;
      std::cout << "New Error Record in " << i << " iteration:" << error << '\n';
    }
    nn.train(LEARNING_RATE / BATCH_SIZE);
  }
}
