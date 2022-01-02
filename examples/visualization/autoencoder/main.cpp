#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>

#include <libkann/layers/Layer.hpp>
#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>

#include <libkann/datasets/MNISTDataSet.hpp>
#include <libkann/datasets/write.hpp>

#include <libkann/Model.hpp>
#include <libkann/Algorithm.hpp>

#include <libkann/utilities/random.hpp>

#include <queue>
#include <mutex>
#include <future>
#include <random>

// These are merely suggestion to the window manager and need not be obeyed
static constexpr unsigned WINDOW_WIDTH = 800;
static constexpr unsigned WINDOW_HEIGHT = 600;

static constexpr size_t FEATURES_COUNT = 64;
static constexpr double LEARNING_RATE = 0.05;

struct Result
{
  sf::Image input;
  sf::Image output;
};

std::mutex resultsLock;
std::vector<Result> results;

static void attachWeightActivationLayers(std::vector<std::shared_ptr<kann::Layer>>& layers, const std::vector<size_t>& topology, kann::ActivationFunction::Type activationType)
{
  const auto activationFunction = kann::ActivationFunction(activationType);
  for(size_t i=0; i < topology.size()-1; ++i)
  {
    size_t prevSize = topology[i];
    size_t nextSize = topology[i+1];
    layers.push_back(std::make_shared<kann::WeightLayer>(prevSize, nextSize));
    layers.push_back(std::make_shared<kann::ActivationLayer>(nextSize, activationFunction));
  }
}

static auto buildAndRunAutoEncoder()
{
  std::default_random_engine engine(random<std::mt19937::result_type>());

  kann::MNISTDataSet trainingDataSet(
    "datasets/mnist/train-images-idx3-ubyte",
    "datasets/mnist/train-labels-idx1-ubyte"
  );

  kann::MNISTDataSet testingDataSet(
    "datasets/mnist/t10k-images-idx3-ubyte",
    "datasets/mnist/t10k-labels-idx1-ubyte"
  );

  std::vector<std::shared_ptr<kann::Layer>> encoderLayers;
  attachWeightActivationLayers(encoderLayers, {kann::MNISTDataSet::IMAGE_SIZE, 256, FEATURES_COUNT}, kann::ActivationFunction::Type::SIGMOID);
  for(auto& layer : encoderLayers)
    layer->randomize(engine);

  std::vector<std::shared_ptr<kann::Layer>> decoderLayers;
  attachWeightActivationLayers(decoderLayers, {FEATURES_COUNT, 256, kann::MNISTDataSet::IMAGE_SIZE}, kann::ActivationFunction::Type::SIGMOID);
  for(auto& layer : decoderLayers)
    layer->randomize(engine);

  auto [autoEncoderModel, decoderModel] = kann::buildSimpleAutoEncoderModel(std::move(encoderLayers), std::move(decoderLayers));
  kann::train(autoEncoderModel, trainingDataSet, kann::MNISTDataSet::COLUMN_IMAGE, kann::MNISTDataSet::COLUMN_IMAGE, LEARNING_RATE, [](kann::Info info){
    std::lock_guard lockGuard(resultsLock);
    results.push_back(Result{
      .input  = kann::toImage(info.model.input(),  kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH),
      .output = kann::toImage(info.model.output(), kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH)
    });
  });
}

int main()
{
  auto thrd = std::thread(buildAndRunAutoEncoder);
  thrd.detach();

  sf::RenderWindow window;
  window.create(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "AutoEncoder Visualization");

  sf::View view(sf::FloatRect(0.0f, 0.0f, 1.0f, 1.0f));
  window.setView(view);

  sf::Texture input, output;
  size_t counter = 0;

  while(window.isOpen())
  {
    sf::Event evnt;
    while(window.pollEvent(evnt))
      switch(evnt.type)
      {
      case sf::Event::Closed:
        window.close();
        break;
      default:
        break;
      }

    // Update texture
    if(counter++ % 1024 == 0)
    {
      std::lock_guard lockGuard(resultsLock);
      if(!results.empty())
      {
        input.loadFromImage(results.back().input);
        output.loadFromImage(results.back().output);
      }
    }

    window.clear();

    sf::Sprite sprite;

    sprite.setTexture(input);
    sprite.setPosition(0.0f, 0.0f);
    sprite.setScale(0.5f / sprite.getLocalBounds().width, 1.0f / sprite.getLocalBounds().height);
    window.draw(sprite);

    sprite.setTexture(output);
    sprite.setPosition(0.5f, 0.0f);
    sprite.setScale(0.5f / sprite.getLocalBounds().width, 1.0f / sprite.getLocalBounds().height);
    window.draw(sprite);

    window.display();
  }
}
