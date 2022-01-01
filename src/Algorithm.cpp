#include <libkann/Algorithm.hpp>
#include <limits>

namespace kann
{
  template<typename... Args>
  static void showProgressBar(std::string_view name, size_t count, size_t total, const Args&... args)
  {
    constexpr static size_t width = 40;
    std::cout << "\e[?25l";

    std::cout << name << "[";
    for(size_t i=0; i<width; ++i)
      if((float)i/width < (float)count/total)
        std::cout << "=";
      else
        std::cout << " ";

    std::cout << " - ";
    std::cout << count << "/" << total;
    std::cout << "]";

    (void)(std::cout << ... << args);

    std::cout << "\r";

    std::cout << "\e[?25h";

    if(count+1==total)
      std::cout << std::endl;
  }

  Callback defaultCallback(std::string_view name)
  {
    return [=](Info info){
      showProgressBar(name, info.i, info.size, "Cost:", info.cost);
    };
  }

  void train(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, float learningRate, Callback callback)
  {
    const auto size = dataSet.size();

    Eigen::VectorXd input, expectedOutput;
    for(size_t i=0; i<size; ++i)
    {
      dataSet.get(inputColumn,  i, input);
      dataSet.get(outputColumn, i, expectedOutput);
      model.feedForward(input);
      model.backPropagate(expectedOutput);
      model.train(learningRate);

      callback(Info{
        .model = model,
        .i = i,
        .size = size,
        .input = input,
        .output = model.output(),
        .expectedOutput = expectedOutput,
        .cost = model.cost(expectedOutput)
      });
    }
  }

  double test(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, Callback callback)
  {
    const auto size = dataSet.size();

    double correctness = 0.0;
    Eigen::VectorXd input, expectedOutput;
    for(size_t i=0; i<size; ++i)
    {
      dataSet.get(inputColumn,  i, input);
      dataSet.get(outputColumn, i, expectedOutput);

      model.feedForward(input);
      correctness += dataSet.correctness(outputColumn, i, model.output());

      callback(Info{
        .model = model,
        .i = i,
        .size = size,
        .input = input,
        .output = model.output(),
        .expectedOutput = expectedOutput,
        .cost = model.cost(expectedOutput)
      });
    }
    correctness /= size;
    return correctness;
  }
}
