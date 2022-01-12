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

#include <libkann/FunctionalModel.hpp>
#include <libkann/Algorithm.hpp>

#include <libkann/utilities/random.hpp>

#include <atomic>
#include <mutex>
#include <thread>
#include <random>
#include <filesystem>
#include <iomanip>

#include <getopt.h>

// These are merely suggestion to the window manager and need not be obeyed
static constexpr unsigned WINDOW_WIDTH = 800;
static constexpr unsigned WINDOW_HEIGHT = 600;

static constexpr size_t FEATURES_COUNT = 64;
static constexpr double LEARNING_RATE = 0.05;

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

class Runner
{
public:
  Runner(std::filesystem::path outputDirectory)
    : m_outputDirectory(std::move(outputDirectory)) {}

public:
  struct State
  {
    std::string_view label;

    size_t i;
    size_t size;

    sf::Image input;
    sf::Image output;
  };

public:
  ~Runner()
  {
    m_stop.store(true);
    if(m_worker.joinable())
      m_worker.join();
  }

public:
  void run()
  {
    m_stop.store(false);
    m_worker = std::thread(&Runner::_run, this);
  }

private:
  void _run()
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

    const char* label;
    auto callback = [this, &label](kann::Info info){
      auto inputImage  = kann::toImage(info.model.input(),  kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);
      auto outputImage = kann::toImage(info.output, kann::MNISTDataSet::IMAGE_WIDTH, kann::MNISTDataSet::IMAGE_WIDTH);

      // Update state
      {
        std::lock_guard lockGuard(m_lock);
        m_state = State{
          .label  = label,
          .i      = info.i,
          .size   = info.size,
          .input  = inputImage,
          .output = outputImage
        };
      }

      // Write output
      std::ostringstream fileName;
      fileName << std::setfill('0') << std::setw(std::ceil(std::log10(info.size))) << info.i << ".png";
      outputImage.saveToFile(m_outputDirectory / label / fileName.str());

      return !m_stop.load();
    };

    auto [autoEncoderModel, decoderModel] = kann::buildSimpleAutoEncoderModel(std::move(encoderLayers), std::move(decoderLayers));

    label = "Training";

    std::filesystem::create_directories(m_outputDirectory / label);
    kann::train(*autoEncoderModel, trainingDataSet, kann::MNISTDataSet::COLUMN_IMAGE, kann::MNISTDataSet::COLUMN_IMAGE, LEARNING_RATE, callback);

    label = "Testing";

    std::filesystem::create_directories(m_outputDirectory / label);
    kann::run(*autoEncoderModel, testingDataSet, kann::MNISTDataSet::COLUMN_IMAGE, callback);
  }

public:
  std::optional<State> state() const
  {
    std::lock_guard lockGuard(m_lock);
    return m_state;
  }

private:
  std::filesystem::path m_outputDirectory;

private:
  mutable std::mutex m_lock;
  std::optional<State> m_state;

private:
  std::atomic<bool> m_stop;
  std::thread m_worker;
};

void usage()
{
  std::clog << "Usage: autoencoder --output OUTPUT_DIRECTORY\n";
  std::clog << "\n";

  std::clog << "Options:\n";
  std::clog << "  -o,--output\n";
  std::clog << "    specify output directory\n";
  std::clog << "  -h,--help\n";
  std::clog << "    print this help message\n";
}

int main(int argc, char* argv[])
{
  // Option Parsing
  int opt;
  struct option options[] =
  {
    {"output", required_argument, nullptr, 'o' },
    {"help"                      , no_argument      , nullptr, 'h'},
    {0, 0, 0, 0}
  };

  std::optional<std::string> outputDirectory;

  int c;
  int indexptr;
  while((c = getopt_long(argc, argv, "o:h", options, &indexptr)) != -1)
    switch(c)
    {
    case 'o':
      if(optarg)
        outputDirectory = optarg;

      break;
    case 'h':
      usage();
      return EXIT_SUCCESS;
    case '?':
      usage();
      return EXIT_FAILURE;
  }

  if(optind < argc)
  {
    std::clog << "error: too many arguments\n";
    usage();
    return EXIT_FAILURE;
  }

  while((opt = getopt_long(argc, argv, "o:", options, NULL)) != -1)
    if(opt == 'o')
      outputDirectory = optarg;

  if(!outputDirectory)
  {
    std::cerr << "error: no output directory\n";
    return -1;
  }

  // Main Program
  Runner runner(*outputDirectory);
  runner.run();

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
      const auto state = runner.state();
      if(state)
      {
        text.setString(std::string(state->label) + ":" + std::to_string(state->i) + "/" + std::to_string(state->size));
        input.loadFromImage(state->input);
        output.loadFromImage(state->output);
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
