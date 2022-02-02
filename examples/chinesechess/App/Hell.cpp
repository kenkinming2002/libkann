#include "Hell.hpp"

#include "../utilities/lexical_cast.hpp"

#include <libkann/layers/SequentialLayer.hpp>
#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>

#include <libkann/utilities/random.hpp>

#include <libkann/Build.hpp>

#include <SFML/Window/Event.hpp>

#include <iostream>
#include <cstdlib>
#include <getopt.h>

static constexpr unsigned WINDOW_WIDTH = 1000;
static constexpr unsigned WINDOW_HEIGHT = 1000;

namespace App
{
  void Hell::usage()
  {
    std::clog << "Usage: chinesechess hell [OPTIONS]\n";
    std::clog << "\n";

    std::clog << "A AI that is unreasonably stupid at first but will learn from you and mutate. Its only purpose is to defeat you.\n";
    std::clog << "\n";

    std::clog << "Options:\n";
    std::clog << "  --agent-seed\n";
    std::clog << "    specify seed for the agent [DEFAULT: Random]\n";
    std::clog << "  --agent-hidden-layers\n";
    std::clog << "    specify topology of hidden layers for agent [DEFAULT: 32,32,32,32]\n";
    std::clog << "  --agent-learning-rate\n";
    std::clog << "    specify learning rate for agent [DEFAULT: 0.1]\n";
    std::clog << "  -h,--help\n";
    std::clog << "    print this help message\n";
  }

  int Hell::main(int argc, char* argv[])
  {
    // Program options
    std::optional<seed_type> seed;
    std::vector<size_t> agentHiddenLayers;
    std::optional<double> agentLearningRate;

    // Parse Program Options
    enum OptionVal : int { SEED = 0, AGENT_HIDDEN_LAYERS = 1, AGENT_LEARNING_RATE = 2 };

    struct option options[] =
    {
      {"seed"               , required_argument, nullptr, SEED},
      {"agent-hidden-layers", required_argument, nullptr, AGENT_HIDDEN_LAYERS},
      {"agent-learning-rate", required_argument, nullptr, AGENT_LEARNING_RATE},
      {"help"               , no_argument      , nullptr, 'h'},
      {0, 0, 0, 0}
    };
    int c;
    int indexptr;
    while((c = getopt_long(argc, argv, "oh", options, &indexptr)) != -1)
      switch(c)
      {
      case SEED:
        seed = lexical_cast<seed_type>(optarg);
        break;
        break;
      case AGENT_HIDDEN_LAYERS:
        for(char* cur = strtok(optarg, ","); cur != nullptr; cur = strtok(nullptr, ","))
          agentHiddenLayers.push_back(lexical_cast<size_t>(cur));
        break;
      case AGENT_LEARNING_RATE:
        agentLearningRate = lexical_cast<double>(optarg);
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

  if(!seed) seed = random<seed_type>();
  if(agentHiddenLayers.empty()) agentHiddenLayers = {32, 32, 32, 32};
  if(!agentLearningRate) agentLearningRate = 0.1;

  // DEBUG
  {
    std::clog << "DEBUG: seed=" << *seed << '\n';
    std::clog << "DEBUG: agentHiddenLayers={";
    for(size_t i=0; i<agentHiddenLayers.size()-1; ++i)
      std::clog << agentHiddenLayers[i] << ',';
    std::clog << agentHiddenLayers.back();
    std::clog << "}\n";
    std::clog << "DEBUG: agentLearningRate=" << *agentLearningRate << '\n';
  }

  Hell(*seed, agentHiddenLayers, *agentLearningRate).run();
  return EXIT_SUCCESS;
  }

  namespace
  {
    Agent makeAgent(Hell::seed_type seed, const std::vector<size_t>& agentHiddenLayers, double learningRate)
    {
      Hell::random_engine_type engine(seed);

      static std::shared_ptr<kann::Layer> layer = [&agentHiddenLayers]()
      {
        const auto activationFunction = kann::ActivationFunction(kann::ActivationFunction::Type::SIGMOID);

        std::vector<size_t> topology;

        topology.push_back(Agent::INPUT_LAYER_SIZE);
        topology.insert(topology.end(), agentHiddenLayers.begin(), agentHiddenLayers.end());
        topology.push_back(Agent::OUTPUT_LAYER_SIZE);

        auto layer = std::make_shared<kann::SequentialLayer>();
        for(size_t i=0; i < topology.size()-1; ++i)
        {
          size_t prevSize = topology[i];
          size_t nextSize = topology[i+1];
          layer->addLayer(std::make_shared<kann::WeightLayer>(prevSize, nextSize));
          layer->addLayer(std::make_shared<kann::ActivationLayer>(nextSize, activationFunction));
        }
        return layer;
      }();

      auto model = std::make_shared<kann::Model>(layer);
      model->randomize();
      return Agent(std::move(model), learningRate);
    }
  }

  Hell::Hell(seed_type seed, const std::vector<size_t>& agentHiddenLayers, double agentLearningRate)
    : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), ""),
      m_renderer(m_window),
      m_agent(makeAgent(seed, agentHiddenLayers, agentLearningRate)) {}

  void Hell::run()
  {
    while(this->loop());
  }

  bool Hell::loop()
  {
    this->handleInput();
    this->update();
    this->render();
    return m_window.isOpen();
  }

  void Hell::handleInput()
  {
    sf::Event event;
    while (m_window.pollEvent(event))
    {
      if(m_human.handleInput(m_renderer, m_game.board(), Board::Cell::Color::RED, event))
        continue;

      if(event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::U)
      {
        m_game.undoMove();
        m_game.undoMove();
      }

      if(event.type == sf::Event::Closed)
        m_window.close();
    }
  }

  void Hell::update()
  {
    // Check
    if(m_game.ended())
    {
      m_agent.learnFrom(m_game);
      m_game = Game();
    }

    // Human
    if(auto move = m_human.pollMove(m_game.board(), Board::Cell::Color::RED))
      m_game.performMove(*move, Board::Cell::Color::RED);
    else
      return;

    // Check
    if(m_game.ended())
      return;

    // Agent
    if(auto move = m_agent.selectMove(m_game.board(), Board::Cell::Color::BLACK))
      m_game.performMove(*move, Board::Cell::Color::BLACK);
    else
      m_game.moveExhausted(Board::Cell::Color::BLACK);
  }

  void Hell::render()
  {
    m_renderer.draw(m_game.board());
    m_window.display();
  }
}
