#pragma once

#include <SFML/Graphics/RenderWindow.hpp>

#include <filesystem>
#include "Population.hpp"
#include "Renderer.hpp"

#include "Agent.hpp"
#include "Human.hpp"

class AppGeneratePopulations
{
public:
  using random_engine_type = Population::random_engine_type;
  using seed_type = Population::seed_type;

public:
  AppGeneratePopulations(const char* outputDirectory, seed_type seed);

public:
  void run();

private:
  std::filesystem::path m_outputDirectory;
  Population m_population;
};

class AppMatchAgents
{
public:
  AppMatchAgents(const char* agent1FilePath, const char* agent2FilePath);

public:
  void run();

private:
  Agent m_agent1, m_agent2;
};

class AppMatchAgentPlayer
{
public:
  AppMatchAgentPlayer(const char* agentFilePath);

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
  bool m_matchEnded = false;

  Board m_board;
  Agent m_agent;
  Human m_human;
};
