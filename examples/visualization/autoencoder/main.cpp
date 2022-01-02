#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Text.hpp>

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

struct State
{
  std::mutex lock;

  std::string_view label;

  std::vector<Result> results;
  size_t i;
  size_t size;
};

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

static auto buildAndRunAutoEncoder(State& state)
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
  kann::train(autoEncoderModel, trainingDataSet, kann::MNISTDataSet::COLUMN_IMAGE, kann::MNISTDataSet::COLUMN_IMAGE, LEARNING_RATE, [&state](kann::Info info){
    std::lock_guard lockGuard(state.lock);

    state.results.push_back(Result{
      .input  = kann::toImage(info.model.input(),  kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH),
      .output = kann::toImage(info.model.output(), kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH)
    });
    state.label = "Training";
    state.i    = info.i;
    state.size = info.size;
  });

  kann::run(autoEncoderModel, testingDataSet, kann::MNISTDataSet::COLUMN_IMAGE, [&state](kann::Info info){
    std::lock_guard lockGuard(state.lock);

    state.results.push_back(Result{
      .input  = kann::toImage(info.model.input(),  kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH),
      .output = kann::toImage(info.model.output(), kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH)
    });
    state.label = "Testing";
    state.i    = info.i;
    state.size = info.size;
  });
}

int main()
{
  State state;
  std::thread(buildAndRunAutoEncoder, std::ref(state)).detach();

  sf::RenderWindow window;
  window.create(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "AutoEncoder Visualization");

  sf::Font font;
  sf::Text text;
  sf::Texture input, output;

  if(!font.loadFromFile("resources/fonts/NotoSansMono-Regular.ttf"))
    throw std::runtime_error("Failed to load font");

  text.setFont(font);

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
      std::lock_guard lockGuard(state.lock);
      text.setString(std::string(state.label) + ":" + std::to_string(state.i) + "/" + std::to_string(state.size));
      if(!state.results.empty())
      {
        input.loadFromImage(state.results.back().input);
        output.loadFromImage(state.results.back().output);
      }
    }

    window.clear();

    // Result
    {
      sf::View view(sf::FloatRect(0.0f, 0.0f, 1.0f, 1.0f));
      window.setView(view);

      sf::Sprite sprite;

      sprite.setTexture(input);
      sprite.setPosition(0.0f, 0.0f);
      sprite.setScale(0.5f / sprite.getLocalBounds().width, 1.0f / sprite.getLocalBounds().height);
      window.draw(sprite);

      sprite.setTexture(output);
      sprite.setPosition(0.5f, 0.0f);
      sprite.setScale(0.5f / sprite.getLocalBounds().width, 1.0f / sprite.getLocalBounds().height);
      window.draw(sprite);
    }

    // Text
    {
      auto windowSize = window.getSize();
      sf::View view(sf::FloatRect(0.0f, 0.0f, windowSize.x, windowSize.y));
      window.setView(view);

      window.draw(text);
    }

    window.display();
  }
}
