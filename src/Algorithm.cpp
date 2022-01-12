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
      return true;
    };
  }

  GANCallback defaultGANCallback(std::string_view name)
  {
    return [=](GANInfo info){
      showProgressBar(name, info.i, info.size, " ",
        "GAN Output(Fake image):",           info.GANOutput(0), ", ",
        "Discriminator Output(Real image):", info.discriminatorOutput(0)
      );
      return true;
    };
  }

  void run(Model& model, const DataSet& dataSet, size_t column, Callback callback)
  {
    const auto size = dataSet.size();

    Eigen::VectorXd input;
    for(size_t i=0; i<size; ++i)
    {
      dataSet.get(column,  i, input);

      model.input(input);
      Eigen::VectorXd output = model.feedForward();

      const bool result = callback(Info{
        .model = model,
        .i = i,
        .size = size,
        .output = output
      });
      if(!result)
        break;
    }
  }

  void train(Model& model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, float learningRate, Callback callback)
  {
    const auto size = dataSet.size();

    Eigen::VectorXd input, expectedOutput;
    for(size_t i=0; i<size; ++i)
    {
      dataSet.get(inputColumn,  i, input);
      dataSet.get(outputColumn, i, expectedOutput);

      model.input(input);
      Eigen::VectorXd output = model.feedForward();
      Eigen::VectorXd outputGradient = (output-expectedOutput) * 2.0;

      model.outputGradient(outputGradient);
      Eigen::VectorXd inputGradient = model.backPropagate();

      model.train(learningRate);

      const bool result = callback(Info{
        .model = model,
        .i = i,
        .size = size,
        .expectedOutput = expectedOutput,
        .cost = outputGradient.squaredNorm()
      });
      if(!result)
        break;
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

      model.input(input);
      Eigen::VectorXd output = model.feedForward();
      Eigen::VectorXd outputGradient = (output-expectedOutput) * 2.0;

      correctness += dataSet.correctness(outputColumn, i, output);

      const bool result = callback(Info{
        .model = model,
        .i = i,
        .size = size,
        .expectedOutput = expectedOutput,
        .cost = outputGradient.squaredNorm()
      });

      if(!result)
        break;
    }
    correctness /= size;
    return correctness;
  }

  void trainGAN(Model& GANModel, Model& generatorModel, Model& discriminatorModel,
      const DataSet& dataSetLatent, const DataSet& dataSet,
      size_t columnLatent, size_t column,
      float learningRate, GANCallback callback)
  {
    // TODO: Support if their size are not equal
    assert(dataSetLatent.size() == dataSet.size());

    const auto size = dataSetLatent.size();

    Eigen::VectorXd input;
    Eigen::VectorXd expectedOutput(1);
    for(size_t i=0; i<size; ++i)
    {
      // Train on latent data set
      dataSetLatent.get(columnLatent, i, input);
      GANModel.input(input);
      Eigen::VectorXd GANOutput = GANModel.feedForward();
      {
        expectedOutput(0) = 1.0;
        Eigen::VectorXd outputGradient = (GANOutput-expectedOutput) * 2.0;
        GANModel.outputGradient(outputGradient);
        GANModel.backPropagate();
        GANModel.train(learningRate * 0.5, TAG_GAN_GENERATOR);
      }

      {
        expectedOutput(0) = 0.0;
        Eigen::VectorXd outputGradient = (GANOutput-expectedOutput) * 2.0;
        GANModel.outputGradient(outputGradient);
        GANModel.backPropagate();
        GANModel.train(learningRate * 0.5, TAG_GAN_DISCRIMINATOR);
      }

      /* Sample the generator
       * TODO: Optimize we are redoing the calculation done
       *       when we call GANModel.feedForward(input) */
      generatorModel.input(input);
      Eigen::VectorXd generatorOutput = generatorModel.feedForward();


      // Train on real data set
      dataSet.get(column, i, input);
      discriminatorModel.input(input);
      Eigen::VectorXd discriminatorOutput = discriminatorModel.feedForward();
      {
        expectedOutput(0) = 1.0;
        Eigen::VectorXd outputGradient = (discriminatorOutput-expectedOutput) * 2.0;
        discriminatorModel.outputGradient(outputGradient);
        discriminatorModel.backPropagate();
        discriminatorModel.train(learningRate * 0.5);
      }

      const bool result = callback(GANInfo{
        .GANModel = GANModel,
        .generatorModel     = generatorModel,
        .discriminatorModel = discriminatorModel,
        .i = i,
        .size = size,
        .GANOutput = GANOutput,
        .generatorOutput = generatorOutput,
        .discriminatorOutput = discriminatorOutput
      });
      if(!result)
        break;

    }
  }
}
