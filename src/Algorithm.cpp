#include <libkann/Algorithm.hpp>

#include <libkann/Optimizer.hpp>
#include <libkann/Predictor.hpp>

#include <limits>

namespace kann
{
  /* Note: In the implementation, we have the following pattern
   * ```
   *  auto info = Info{...};
   *  co_yield info;
   * ```
   *
   * In reality, we should be able to do directly
   * ```
   *  co_yield Info{...};
   * ```
   *
   * but that leads to double-free/use-after-free error in GCC as reported in
   * GCC Bug 103909 - co_yield of aggregate-initialized temporaries leads to
   * segmentation faults(https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103909)
   */
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

  void displayInfo(std::string_view name, const Info& info)
  {
    showProgressBar(name, info.i, info.size, "Cost:", info.cost);
  }

  void displayInfo(std::string_view name, const GANInfo& info)
  {
    showProgressBar(name, info.i, info.size, " ",
      "GAN Output(Fake image):",           info.GANOutput->asArray()(0), ", ",
      "Discriminator Output(Real image):", info.discriminatorOutput->asArray()(0)
    );
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

  Task<void, Info> run(std::shared_ptr<NewModel> model,
      std::vector<std::shared_ptr<const Tensor>> inputs)
  {
    for(size_t i=0; i<inputs.size(); ++i)
    {
      auto output = model->predict(inputs[i]);
      auto info = Info{
        .model = model,
        .i = i,
        .size = inputs.size(),
        .input  = std::move(inputs[i]),
        .output = std::move(output)
      };
      co_yield info;
    }
  }

  Task<void, Info> train(std::shared_ptr<NewModel> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs,
      double learningRate, size_t batchSize)
  {
    assert(inputs.size() == expectedOutputs.size());
    const size_t size = inputs.size() / batchSize * batchSize;

    for(size_t i=0; i<size; i+=batchSize)
    {
      auto inputsBatch          = std::vector<std::shared_ptr<const Tensor>>(&inputs[i],          &inputs[i+batchSize]);
      auto expectedOutputsBatch = std::vector<std::shared_ptr<const Tensor>>(&expectedOutputs[i], &expectedOutputs[i+batchSize]);

      auto [outputsBatch, costs] = model->optimize(learningRate, NEW_TAG_ALL, inputsBatch, expectedOutputsBatch);

      assert(inputsBatch.size()          == batchSize);
      assert(expectedOutputsBatch.size() == batchSize);
      assert(outputsBatch.size()         == batchSize);
      assert(costs.size()                == batchSize);

      for(size_t j=0; j<batchSize; ++j)
      {
        auto info = Info{
          .model = model,
          .i = i,
          .size = size,
          .input          = std::move(inputsBatch[j]),
          .output         = std::move(outputsBatch[j]),
          .expectedOutput = std::move(expectedOutputsBatch[j]),
          .cost = costs[j]
        };
        co_yield info;
      }
    }
  }

  Task<void, GANInfo> trainGAN(std::shared_ptr<NewModel> model,
      std::shared_ptr<NewModel> generatorModel,
      std::shared_ptr<NewModel> discriminatorModel,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> latentInputs,
      double learningRate, size_t batchSize)
  {
    assert(latentInputs.size() == inputs.size());
    const size_t size = inputs.size() / batchSize * batchSize;

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
      auto [discriminatorResult, discriminatorCost] = discriminatorModel->optimize(learningRate, NEW_TAG_ALL, inputsBatch, oneBatch);

      // Predict on latent data set
      auto generatorResult = convert(latentInputsBatch, [&generatorModel](const auto& input){ return generatorModel->predict(input);});

      // Train on latent data set
      auto [discriminatorCombinedResult, discriminatorCombinedCost] = model->optimize(learningRate, TAG_GAN_DISCRIMINATOR, latentInputsBatch, zeroBatch);
      auto [generatorCombinedResult,     generatorCombinedCost]     = model->optimize(learningRate, TAG_GAN_GENERATOR,     latentInputsBatch, oneBatch);

      for(size_t j=0; j<batchSize; ++j)
      {
        auto info = GANInfo{
          .GANModel = model,
          .generatorModel = generatorModel,
          .discriminatorModel = discriminatorModel,
          .i = i,
          .size = size,
          .GANOutput           = discriminatorCombinedResult[j],
          .generatorOutput     = generatorResult[j],
          .discriminatorOutput = discriminatorResult[j]
        };
        co_yield info;
      }
    }
  }

  Task<double, Info> test(std::shared_ptr<NewModel> model,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs)
  {
    // How do we inplement correctness
    assert(inputs.size() == expectedOutputs.size());
    const size_t size = inputs.size();
    size_t correct = 0;

    for(size_t i=0; i<size; ++i)
    {
      auto output = model->predict(inputs[i]);

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
      co_yield info;
    }

    co_return (double)correct / size;
  }
}
