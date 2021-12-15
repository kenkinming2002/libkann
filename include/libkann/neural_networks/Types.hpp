#pragma once

#include <Eigen/Eigen>

#include <vector>

namespace kann
{
  struct FeedForwardResult
  {
    std::vector<Eigen::VectorXd> data; // Data between layer, which can be both input and output
    const Eigen::VectorXd& output() const { return data.back(); }
  };

  struct RecurrentFeedForwardResult
  {
    std::vector<Eigen::VectorXd> data; // Data between layer, which can be both input and output

    Eigen::VectorXd memory;
    Eigen::VectorXd output;
  };

  struct BackPropagationResult
  {
    std::vector<Eigen::RowVectorXd> gradients; // Gradients of data between layer
    std::vector<Eigen::ArrayXd> layerGradients; // Gradients for params of layer
  };

}
