#include <libkann/Algorithm.hpp>

#include <libkann/Optimizer.hpp>
#include <libkann/Predictor.hpp>

#include <limits>

namespace kann
{
  template<typename U, typename UnaryFunc, typename T = std::result_of_t<UnaryFunc(const U&)>>
  static std::vector<T> convert(const std::vector<U>& in, const UnaryFunc& f)
  {
    std::vector<T> out;
    out.reserve(in.size());
    for(const auto& v : in)
      out.push_back(f(v));

    return out;
  }

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
        "GAN Output(Fake image):",           info.GANOutput->asArray()(0), ", ",
        "Discriminator Output(Real image):", info.discriminatorOutput->asArray()(0)
      );
      return true;
    };
  }

  std::vector<std::shared_ptr<const Tensor>> load(const DataSet& dataSet, size_t column)
  {
    const size_t size = dataSet.size();

    std::vector<std::shared_ptr<const Tensor>> result;
    result.reserve(dataSet.size());
    for(size_t i=0; i<size; ++i)
      result.push_back(dataSet.get(column, i));

    return result;
  }

  void run(std::shared_ptr<const Model> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      Callback callback)
  {
    Predictor predictor(model);
    for(size_t i=0; i<inputs.size(); ++i)
    {
      auto output = predictor.predict(inputs[i]);
      auto info = Info{
        .model = model,
        .i = i,
        .size = inputs.size(),
        .input  = std::move(inputs[i]),
        .output = std::move(output)
      };
      if(auto result = callback(info); !result)
        return;
    }
  }

  void train(std::shared_ptr<Model> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs,
      double learningRate, size_t batchSize,
      Callback callback)
  {
    assert(inputs.size() == expectedOutputs.size());
    const size_t size = inputs.size() / batchSize * batchSize;

    Optimizer optimizer(model, learningRate, batchSize);
    for(size_t i=0; i<size; i+=batchSize)
    {
      auto inputsBatch          = std::vector<std::shared_ptr<const Tensor>>(&inputs[i],          &inputs[i+batchSize]);
      auto expectedOutputsBatch = std::vector<std::shared_ptr<const Tensor>>(&expectedOutputs[i], &expectedOutputs[i+batchSize]);
      auto result = optimizer.optimize(inputsBatch, expectedOutputsBatch, TAG_ALL);

      for(size_t j=0; j<batchSize; ++j)
      {
        auto info = Info{
          .model = model,
          .i = i,
          .size = size,
          .input = std::move(inputsBatch[j]),
          .output = std::move(result[j].first),
          .expectedOutput = std::move(expectedOutputsBatch[j]),
          .cost = result[j].second
        };
      if(auto result = callback(info); !result)
        return;
      }
    }
  }

  void trainGAN(std::shared_ptr<Model> model,
      std::shared_ptr<Model> generatorModel,
      std::shared_ptr<Model> discriminatorModel,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> latentInputs,
      double learningRate, size_t batchSize,
      GANCallback callback)
  {
    assert(latentInputs.size() == inputs.size());
    const size_t size = inputs.size() / batchSize * batchSize;

    Optimizer optimizer(model, learningRate, batchSize);

    Optimizer discriminatorOptimizer(discriminatorModel, learningRate, batchSize);
    Predictor generatorPredictor(generatorModel);

    auto zero = std::make_shared<const Tensor>(1);
    zero->asArray()(0) = 0.0;

    auto one = std::make_shared<const Tensor>(1);
    one->asArray()(0) = 1.0;

    auto zeroBatch = std::vector(batchSize, zero);
    auto oneBatch  = std::vector(batchSize, one);

    for(size_t i=0; i<size; i+=batchSize)
    {
      auto inputsBatch = std::vector<std::shared_ptr<const Tensor>>(&inputs[i], &inputs[i+batchSize]);
      auto latentInputsBatch = std::vector<std::shared_ptr<const Tensor>>(&latentInputs[i], &latentInputs[i+batchSize]);

      // Train on real data set
      auto discriminatorResult = discriminatorOptimizer.optimize(inputsBatch, oneBatch);

      // Predict on latent data set
      auto generatorResult = convert(latentInputsBatch, [&generatorPredictor](const std::shared_ptr<const Tensor>& input){ return generatorPredictor.predict(input);});

      // Train on latent data set
      auto discriminatorCombinedResult = optimizer.optimize(latentInputsBatch, zeroBatch, TAG_GAN_DISCRIMINATOR);
      auto generatorCombinedResult     = optimizer.optimize(latentInputsBatch, oneBatch,  TAG_GAN_GENERATOR);

      for(size_t j=0; j<batchSize; ++j)
      {
        auto info = GANInfo{
          .GANModel = model,
          .generatorModel = generatorModel,
          .discriminatorModel = discriminatorModel,
          .i = i,
          .size = size,
          .GANOutput           = discriminatorCombinedResult[j].first,
          .generatorOutput     = generatorResult[j],
          .discriminatorOutput = discriminatorResult[j].first,
        };
        if(auto result = callback(info); !result)
          return;
      }
    }
  }

  double test(std::shared_ptr<const Model> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs,
      Callback callback)
  {
    // How do we inplement correctness
    assert(inputs.size() == expectedOutputs.size());
    const size_t size = inputs.size();
    size_t correct = 0;

    Predictor predictor(model);
    for(size_t i=0; i<size; ++i)
    {
      auto output = predictor.predict(inputs[i]);

      size_t index1, index2;
      output->asArray().maxCoeff(&index1);
      expectedOutputs[i]->asArray().maxCoeff(&index2);
      if(index1 == index2)
        ++correct;

      auto cost = (2.0 * (output->asVector() - expectedOutputs[i]->asVector())).squaredNorm();

      auto info = Info{
        .model = model,
        .i = i,
        .size = size,
        .input  = std::move(inputs[i]),
        .output = std::move(output),
        .cost = cost
      };
      if(auto result = callback(info); !result)
        break;
    }

    return (double)correct / size;
  }

}
