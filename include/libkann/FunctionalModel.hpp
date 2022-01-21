#pragma once

#include <libkann/Model.hpp>
#include <libkann/FunctionalVariable.hpp>

#include <memory>

namespace kann
{
  struct FeedBack
  {
    std::shared_ptr<const FunctionalVariable> input, output;

    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(input, output);
    }
  };

  std::shared_ptr<Model> makeFunctionalModel(std::shared_ptr<const FunctionalVariable> input, std::shared_ptr<const FunctionalVariable> output, std::vector<FeedBack> feedBacks = {});
}
