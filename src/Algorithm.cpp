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
  }

  void train(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, float learningRate)
  {
    const auto size = dataSet.size();

    double cost = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd input, expectedOutput;
    for(size_t i=0; i<size; ++i)
    {
      showProgressBar("Training", i, size, "Cost:", cost);
      dataSet.get(inputColumn,  i, input);
      dataSet.get(outputColumn, i, expectedOutput);
      model.feedForward(input);
      model.backPropagate(expectedOutput);
      model.train(learningRate);
      cost = model.cost(expectedOutput);

    }
    std::cout << std::endl;
  }

  double test(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn)
  {
    const auto size = dataSet.size();

    double correctness = 0.0;
    Eigen::VectorXd input;
    for(size_t i=0; i<size; ++i)
    {
      showProgressBar("Testing", i, size);
      dataSet.get(inputColumn,  i, input);
      model.feedForward(input);
      correctness += dataSet.correctness(outputColumn, i, model.output());
    }
    std::cout << std::endl;
    correctness /= size;
    return correctness;
  }
}
