#include "Hell.hpp"

#include "../utilities/lexical_cast.hpp"

#include <libkann/layers/SequentialLayer.hpp>
#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>

#include <libkann/utilities/random.hpp>

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
    std::optional<double> agentLearningRate;

    // Parse Program Options
    enum OptionVal : int { SEED = 0, AGENT_LEARNING_RATE = 1 };

    struct option options[] =
    {
      {"seed"               , required_argument, nullptr, SEED},
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
    if(!agentLearningRate) agentLearningRate = 0.1;

    // DEBUG
    {
      std::clog << "DEBUG: seed=" << *seed << '\n';
      std::clog << "DEBUG: agentLearningRate=" << *agentLearningRate << '\n';
    }

    Hell(*seed, *agentLearningRate).run();
    return EXIT_SUCCESS;
  }

  Hell::Hell(seed_type seed, double agentLearningRate)
    : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), ""),
      m_renderer(m_window),
      m_agent(Agent::makeModel(), agentLearningRate) {}

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
