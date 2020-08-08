#include "App.hpp"

#include <sstream>
#include <iostream>
#include <random>
#include <chrono>

[[gnu::noreturn]] void usage()
{
  std::clog << 
    "Usage: chinesechess COMMANDS\n"
    "       chinesechess generate-populations OUTPUT_DIRECTORY [SEED]\n"
    "       chinesechess match AGENT1 [AGENT2]\n"
    "\n"
    "COMMANDS:\n"
    "  generate-populations    Generate populations and write to OUTPUT_DIRECTORY\n"
    "  match                   Play a match between agent1 and agent2 or player if agent2 is not specified\n"
    "\n";

  std::exit(EXIT_FAILURE);
}

App::GeneratePopulations::seed_type generateSeed()
{
  std::random_device rd;
  if(rd.entropy() != 0.0)
  {
    std::clog << "Generating random seed from std::random_device...\n";
    return rd();
  }
  else
  {
    std::clog << "Generating random seed from current time...\n";
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }
}

auto toSeed(const char* str)
{
  App::GeneratePopulations::seed_type seed;

  std::stringstream ss;
  ss << str;
  ss >> seed;
  if(ss.fail() || !ss.eof())
    usage();

  return seed;
}

int main(int argc, const char* argv[])
{
  if(argc < 3 || argc > 4)
    usage();

  if(strcmp("generate-populations", argv[1]) == 0)
  {
    auto outputDirectory = argv[2];
    auto seed = argc == 3 ? generateSeed() : toSeed(argv[3]);
    App::GeneratePopulations(outputDirectory, seed).run();
  }
  else if(strcmp("match", argv[1]) == 0)
  {
    if(argc == 3)
      App::MatchAgentPlayer(argv[2]).run();
    else
      App::MatchAgents(argv[2], argv[3]).run();
  }
  else
    usage();
}
