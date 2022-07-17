#pragma once

#include <libkann/Optimizer.hpp>
#include <libkann/optimizers/SimpleOptimizer.hpp>
#include <libkann/optimizers/AdamOptimizer.hpp>

#include <libkann/LossFunction.hpp>
#include <libkann/loss_functions/Lp.hpp>
#include <libkann/loss_functions/CrossEntropy.hpp>

#include <stdexcept>

#include <fmt/core.h>

inline std::shared_ptr<kann::Optimizer> create_optimizer(std::string name, std::string arg)
{
  if(name == "simple")
  {
    const float learning_rate = std::stof(arg);
    return std::make_shared<kann::SimpleOptimizer>(learning_rate);
  }
  else if(name == "adam")
  {
    std::stringstream ss(arg);
    auto next = [&]() -> float
    {
      std::string str;
      if(!std::getline(ss, str, ','))
        throw std::runtime_error(fmt::format("Adam optimizer:Invalid parameters:{}", arg));

      return std::stof(str);
    };

    const float alpha = next(), beta1 = next(), beta2 = next(), epsilon = next();
    return std::make_shared<kann::AdamOptimizer>(alpha, beta1, beta2, epsilon);
  }
  else
    throw std::runtime_error(fmt::format("Unknown optimizer name:{}", name));
}

inline std::shared_ptr<kann::LossFunction> create_loss_function(std::string name, std::string arg)
{
  if(name == "lp")
  {
    const unsigned p = std::stoi(arg);
    return std::make_shared<kann::LpLossFunction>(p);
  }
  else if(name == "cross_entropy")
  {
    if(arg != "none")
      throw std::runtime_error(fmt::format("Cross entropy loss function:Invalid arguments:{}:Expected 'none'", arg));

    return std::make_shared<kann::CrossEntropyLossFunction>();
  }
  else
    throw std::runtime_error(fmt::format("Unknown loss function name:{}", name));
}

inline void shuffle(
  std::vector<tensor::Tensor<float>>& images,
  std::vector<tensor::Tensor<float>>& labels,
  auto& prng)
{
  const size_t count = std::min(images.size(), labels.size());
  std::uniform_int_distribution<size_t> dist(0, count-1);
  for(size_t i=0; i<count; ++i)
  {
    size_t index1 = dist(prng), index2 = dist(prng);
    if(index1 == index2)
      continue;

    std::swap(images[index1], images[index2]);
    std::swap(labels[index1], labels[index2]);
  }
}
