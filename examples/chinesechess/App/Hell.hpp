#pragma once

#include <SFML/Graphics/RenderWindow.hpp>
#include <random>

#include "../Game.hpp"
#include "../Renderer.hpp"
#include "../Agent.hpp"
#include "../Human.hpp"

namespace App
{
  class Hell
  {
  public:
    static void usage();
    static int main(int argc, char* argv[]);

  public:
    using random_engine_type = std::default_random_engine;
    using seed_type = random_engine_type::result_type;
    Hell(seed_type seed, const std::vector<size_t>& agentHiddenLayers, double agentLearningRate);

  public:
    void run();

  public:
    bool loop();
    void handleInput();
    void update();
    void render();

  private:
    sf::RenderWindow m_window;
    Renderer m_renderer;

  private:
    Game m_game;
    Agent m_agent;
    Human m_human;
  };
}
