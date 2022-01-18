#include <libkann/Algorithm.hpp>

#include <libkann/Optimizer.hpp>
#include <libkann/Predictor.hpp>

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
        "GAN Output(Fake image):",           info.GANOutput.asArray()(0), ", ",
        "Discriminator Output(Real image):", info.discriminatorOutput.asArray()(0)
      );
      return true;
    };
  }

  void run(std::shared_ptr<Model> model, const DataSet& dataSet, size_t column, Callback callback)
  {
    const auto size = dataSet.size();

    Predictor predictor(model);
    for(size_t i=0; i<size; ++i)
    {
      Tensor input = dataSet.get(column,  i);
      Tensor output = predictor.predict(input);

      const bool result = callback(Info{
        .model = model,
        .i = i,
        .size = size,
        .input = input,
        .output = output
      });
      if(!result)
        break;
    }
  }

  void train(std::shared_ptr<Model> model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, float learningRate, Callback callback)
  {
    const auto size = dataSet.size();

    Optimizer optimizer(model, learningRate);

    for(size_t i=0; i<size; ++i)
    {
      Tensor input          = dataSet.get(inputColumn, i);
      Tensor expectedOutput = dataSet.get(outputColumn, i);

      const auto [output, cost] = optimizer.optimize(input, expectedOutput);

      // TODO: Return real output somehow
      const bool result = callback(Info{
        .model = model,
        .i = i,
        .size = size,
        .input = input,
        .output = output,
        .expectedOutput = expectedOutput,
        .cost = cost
      });
      if(!result)
        break;
    }
  }

  double test(std::shared_ptr<Model> model, const DataSet& dataSet, size_t inputColumn, size_t outputColumn, Callback callback)
  {
    const auto size = dataSet.size();

    double correctness = 0.0;

    Predictor predictor(model);
    for(size_t i=0; i<size; ++i)
    {
      Tensor input          = dataSet.get(inputColumn, i);
      Tensor expectedOutput = dataSet.get(outputColumn, i);

      Tensor output = predictor.predict(input);
      correctness += dataSet.correctness(outputColumn, i, output);

      auto outputGradient = (output.asVector()-expectedOutput.asVector()) * 2.0;
      double cost = outputGradient.squaredNorm();

      const bool result = callback(Info{
        .model = model,
        .i = i,
        .size = size,
        .input = input,
        .output = output,
        .expectedOutput = expectedOutput,
        .cost = cost
      });
      if(!result)
        break;
    }

    correctness /= size;
    return correctness;
  }

  void trainGAN(std::shared_ptr<Model> GANModel, std::shared_ptr<Model> generatorModel, std::shared_ptr<Model> discriminatorModel,
      const DataSet& dataSetLatent, const DataSet& dataSet,
      size_t columnLatent, size_t column,
      float learningRate, GANCallback callback)
  {
    Optimizer GANOptimizer(GANModel, learningRate);

    Predictor generatorPredictor(generatorModel);
    Optimizer discriminatorOptimizer(discriminatorModel, learningRate);

    // TODO: Support if their size are not equal
    assert(dataSetLatent.size() == dataSet.size());
    const auto size = dataSetLatent.size();
    for(size_t i=0; i<size; ++i)
    {
      Tensor input;
      Tensor expectedOutput(1);

      // Train on latent data set
      input = dataSet.get(columnLatent, i);

      // Generator
      expectedOutput.asArray()(0) = 1.0;
      const auto [GANOutput, GANCost] = GANOptimizer.optimize(input, expectedOutput/*, TAG_GENERATOR*/);

      // Discriminator
      expectedOutput.asArray()(0) = 0.0;
      const auto [discriminatorOutput, discriminiatorCost] = GANOptimizer.optimize(input, expectedOutput/*, TAG_DISCRIMINATOR*/);

      // Sample the generator
      const auto generatorOutput = generatorPredictor.predict(input);

      // Train on real data set
      input = dataSet.get(column, i);
      discriminatorOptimizer.optimize(input, expectedOutput);
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
