#include "App.hpp"

#include "Match.hpp"

#include <SFML/Window/Event.hpp>

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cereal/archives/json.hpp>

static constexpr unsigned WINDOW_WIDTH = 1000;
static constexpr unsigned WINDOW_HEIGHT = 1000;

AppGeneratePopulations::AppGeneratePopulations(const char* outputDirectory, seed_type seed) 
  : m_outputDirectory(outputDirectory), m_population(seed, 100) 
{
  std::clog << "Running with seed " << seed << '\n';
}

void AppGeneratePopulations::run()
{
  for(size_t i=0;;++i)
  {
    auto directoryPath = m_outputDirectory / (std::string("population")+std::to_string(i));
    m_population.writeTo(directoryPath);
    m_population.select();
  }
}

namespace
{
  Agent loadAgent(const char* filePath)
  {
    std::ifstream inputFile;
    inputFile.open(filePath);
    if(!inputFile)
      throw std::runtime_error(std::string("Failed to open file ") + filePath);

    cereal::JSONInputArchive inputArchive(inputFile);
    Agent agent;
    inputArchive(agent);
    return agent;
  }
}

AppMatchAgents::AppMatchAgents(const char* agent1FilePath, const char* agent2FilePath)
  : m_agent1(loadAgent(agent1FilePath)), m_agent2(loadAgent(agent2FilePath)) {}

void AppMatchAgents::run()
{
  auto result = match(m_agent1, m_agent2, 1000);
  std::cout << "End result of first game\n";
  std::cout << result.board1;
  std::cout << "End result of second game\n";
  std::cout << result.board2;
  switch(result.winningAgent)
  {
  case PlayerID::NONE:
    std::cout << "Draw\n";
    break;
  case PlayerID::AGENT1:
    std::cout << "First agent won\n";
    break;
  case PlayerID::AGENT2:
    std::cout << "Second agent won\n";
    break;
  default:
    throw std::runtime_error("WTH has happend!");
  }
}

AppMatchAgentPlayer::AppMatchAgentPlayer(const char* agentFilePath) 
  : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "App"),
    m_renderer(m_window),
    m_agent(loadAgent(agentFilePath)) {}

void AppMatchAgentPlayer::run()
{
  while(this->loop());
}

bool AppMatchAgentPlayer::loop()
{
  this->handleInput();
  this->update();
  this->render();
  return m_window.isOpen();
}

void AppMatchAgentPlayer::handleInput()
{
  sf::Event event;
  while (m_window.pollEvent(event))
  {
    if(m_human.handleInput(m_renderer, event))
      continue;

    if(event.type == sf::Event::Closed)
      m_window.close();
  }
}

void AppMatchAgentPlayer::update()
{
  if(m_matchEnded)
    return;

  {
    auto move = m_human.pollMove(m_board, Board::Cell::Color::RED);
    if(!move)
      return;

    if(auto color = m_board.performMove(*move, Board::Cell::Color::RED); color != Board::Cell::Color::NONE)
    {
      m_matchEnded = true;
      return;
    }
  }

  {
    auto move = m_agent.selectMove(m_board, Board::Cell::Color::BLACK);
    if(!move)
    {
      m_matchEnded = true;
      return;
    }

    if(auto color = m_board.performMove(*move, Board::Cell::Color::RED); color != Board::Cell::Color::NONE)
    {
      m_matchEnded = true;
      return;
    }
  }
}

void AppMatchAgentPlayer::render()
{
  m_renderer.draw(m_board);
  m_window.display();
}
