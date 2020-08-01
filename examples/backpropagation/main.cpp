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
  return std::abs(expected - real);
}

template<typename SampleFunc>
void train(NeuralNetwork& nn, size_t batchSize, size_t batchCount, float learningRate, SampleFunc sampleFunc)
{
  std::cout << "Training with learning Rate " << learningRate << "...\n";
  for(size_t i=0; i<batchCount; ++i)
  {
    std::cout << "\riteration " << i+1 << std::flush;
    for(size_t j=0; j<batchSize; ++j)
    {
      auto [input, expectedOutput] = sampleFunc();
      nn.input(static_cast<size_t>(0), input);
      nn.feedForward();

      Eigen::VectorXd expectedOutputVector(1);
      expectedOutputVector(0) = expectedOutput;
      nn.backPropagate(expectedOutputVector);
    }
    nn.train(learningRate);
  }
  std::cout << '\n';
}

template<typename SampleFunc>
void test(NeuralNetwork& nn, size_t count, SampleFunc sampleFunc)
{
  double totalError = 0.0;
  for(size_t i=0; i<count; ++i)
  {
    auto [input, expectedOutput] = sampleFunc();

    nn.input(static_cast<size_t>(0), input);
    nn.feedForward();

    // Log error
    auto output = nn.output(static_cast<size_t>(0));
    auto error = errorFunction(expectedOutput, output);
    totalError += error;
  }
  std::cout << "Error:" << totalError / count << '\n';
}

static constexpr size_t TESINT_SAMPLE_COUNT = 1000;

static constexpr size_t BATCH_COUNT = 1000;
static constexpr size_t BATCH_SIZE = 8;

static constexpr double INITIAL_LEARNING_RATE = 0.01;
static constexpr double LEARNING_RATE_PERSISTENCY = 0.99;

int main()
{
  std::mt19937 generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> inputDist(0.0, M_PI);

  std::array<std::pair<double, double>, TESINT_SAMPLE_COUNT> testingSample;
  for(auto& sample: testingSample)
  {
    sample.first = inputDist(generator);
    sample.second = testFunction(sample.first);
  }

  NeuralNetwork nn({1, 20, 20, 20, 1}, generator);
  for(double learningRate = INITIAL_LEARNING_RATE;; learningRate *= LEARNING_RATE_PERSISTENCY)
  {
    size_t i = 0;
    test(nn, TESINT_SAMPLE_COUNT, [&](){
        return testingSample[i++];
    });

    train(nn, BATCH_SIZE, BATCH_COUNT, learningRate, [&](){
      auto input = inputDist(generator);
      auto expectedOutput = testFunction(input);
      return std::make_pair(input, expectedOutput);
    });
  }
}
