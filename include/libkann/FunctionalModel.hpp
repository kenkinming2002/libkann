#pragma once

#include <libkann/Model.hpp>
#include <libkann/OldVariable.hpp>

#include <memory>

namespace kann
{
  struct FeedBack
  {
    std::shared_ptr<const OldVariable> input, output;

    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(input, output);
    }
  };

  std::shared_ptr<Model> makeFunctionalModel(std::shared_ptr<const OldVariable> input, std::shared_ptr<const OldVariable> output, std::vector<FeedBack> feedBacks = {});
}
