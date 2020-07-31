#include <libkann/NeuralNetwork.hpp>

#include <cmath>
#include <random>
#include <chrono>

#include <iostream>

double testFunction(double i)
{
  return std::sin(i);
}

double errorFunction(double expected, double real)
{
  auto difference = expected - real;
  return std::sqrt(difference * difference);
}

int main()
{
  std::mt19937 generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> inputDist(-20.0f, 20.0f);

  auto input = inputDist(generator);
  auto expectedOutput = testFunction(input);

  for(double minimumError = std::numeric_limits<double>::infinity();;)
  {
    NeuralNetwork nn({1, 5, 5, 1}, generator);
    nn.input(static_cast<size_t>(0), input);
    nn.feedForward();
    auto output = nn.output(static_cast<size_t>(0));
    auto error = errorFunction(expectedOutput, output);
    if(error < minimumError)
    {
      minimumError = error;
      std::cout << "Error:" << error << '\n';
    }
  }
}
