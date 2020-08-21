#include "App.hpp"

#include <libkann/utilities/random.hpp>

#include <sstream>
#include <iostream>
#include <optional>

#include <getopt.h>

template<typename Int>
Int lexical_cast(const char* str)
{
  Int i;

  std::stringstream ss;
  ss << str;
  ss >> i;
  if(ss.fail() || !ss.eof())
    throw std::runtime_error(std::string("Invalid argument ") + str);

  return i;
}

void usageGeneratePopulations()
{
  std::clog << "Usage: chinesechess generate-populations [OPTIONS]\n";
  std::clog << "\n";

  std::clog << "Generate populations from given parameter with genetic algorithm\n";
  std::clog << "\n";

  std::clog << "Options:\n";
  std::clog << "  -o,--output\n";
  std::clog << "    specify output directory\n";
  std::clog << "  --population-seed\n";
  std::clog << "    specify seed for the population [DEFAULT: Random]\n";
  std::clog << "  --population-size\n";
  std::clog << "    specify size of the population [DEFAULT: 10]\n";
  std::clog << "  --agents-hidden-layers\n";
  std::clog << "    specify topology of hidden layers for agents [DEFAULT: 32,32,32,32]\n";
  std::clog << "  --selection-iterations-count\n";
  std::clog << "    specify iterations count in each selection [DEFAULT: 10]\n";
  std::clog << "  -h,--help\n";
  std::clog << "    print this help message\n";
  std::clog << "\n";

  std::clog << "Examples: TODO\n";
}

int mainGeneratePopulations(int argc, char* argv[])
{
  // Program options
  const char* outputDirectory = nullptr;
  std::optional<App::GeneratePopulations::seed_type> populationSeed;
  std::optional<size_t> populationSize;
  std::vector<size_t> agentsHiddenLayers;
  std::optional<size_t> selectionIterationsCount;

  // Parse Program Options
  enum OptionVal : int
  {
    POPULATION_SEED = 0,
    POPULATION_SIZE = 1,
    AGENTS_HIDDEN_LAYERS = 2,
    SELECTION_ITERATIONS_COUNT = 3
  };
  
  struct option options[] = 
  {
    {"output"                    , required_argument, nullptr, 'o'}, 
    {"population-seed"           , required_argument, nullptr, POPULATION_SEED}, 
    {"population-size"           , required_argument, nullptr, POPULATION_SIZE}, 
    {"agents-hidden-layers"      , required_argument, nullptr, AGENTS_HIDDEN_LAYERS}, 
    {"selection-iterations-count", required_argument, nullptr, SELECTION_ITERATIONS_COUNT}, 
    {"help"                      , no_argument      , nullptr, 'h'}, 
    {0, 0, 0, 0}
  };
  int c;
  int indexptr;
  while((c = getopt_long(argc, argv, "oh", options, &indexptr)) != -1)
    switch(c)
    {
    case 'o':
      outputDirectory = optarg;
      break;
    case POPULATION_SEED:
      populationSeed = lexical_cast<App::GeneratePopulations::seed_type>(optarg);
      break;
    case POPULATION_SIZE:
      populationSize = lexical_cast<size_t>(optarg);
      break;
    case AGENTS_HIDDEN_LAYERS:
      for(char* cur = strtok(optarg, ","); cur != nullptr; cur = strtok(nullptr, ","))
        agentsHiddenLayers.push_back(lexical_cast<size_t>(cur));
      break;
    case SELECTION_ITERATIONS_COUNT:
      selectionIterationsCount = lexical_cast<size_t>(optarg);
      break;
    case 'h':
      usageGeneratePopulations();
      return EXIT_SUCCESS;
    case '?':
      usageGeneratePopulations();
      return EXIT_FAILURE;
    }

  if(optind < argc)
  { 
    std::clog << "error: too many arguments\n";
    usageGeneratePopulations(); 
    return EXIT_FAILURE; 
  }

  if(!outputDirectory) 
  { 
    std::clog << "error: output directory not specified\n";
    usageGeneratePopulations(); 
    return EXIT_FAILURE; 
  }
  if(!populationSeed) populationSeed = random<App::GeneratePopulations::seed_type>();
  if(!populationSize) populationSize = 5000;
  if(agentsHiddenLayers.empty()) agentsHiddenLayers = {32, 32, 32, 32};
  if(!selectionIterationsCount) selectionIterationsCount = 10;

  // DEBUG
  {
    std::clog << "DEBUG: outputDirectory=" << (outputDirectory ? outputDirectory : "") << '\n';
    std::clog << "DEBUG: populationSeed=" << *populationSeed << '\n';
    std::clog << "DEBUG: populationSize=" << *populationSize << '\n';
    std::clog << "DEBUG: agentsHiddenLayers={";
    for(size_t i=0; i<agentsHiddenLayers.size()-1; ++i)
      std::clog << agentsHiddenLayers[i] << ',';
    std::clog << agentsHiddenLayers.back();
    std::clog << "}\n";
    std::clog << "DEBUG: selectionIterationsCount=" << *selectionIterationsCount << '\n';
  }

  App::GeneratePopulations(outputDirectory, *populationSeed, *populationSize, agentsHiddenLayers, *selectionIterationsCount).run();
  return EXIT_SUCCESS;
}

void usageMatch()
{
  std::clog << "Usage: chinesechess match AGENT1 [AGENT2]\n";
  std::clog << "\n";

  std::clog << "Match between the 2 agents if AGENT2 is specified. Otherwise, the user will play against AGENT1.\n";
  std::clog << "\n";

  std::clog << "Options:\n";
  std::clog << "  -h,--help\n";
  std::clog << "    print this help message\n";
}

int mainMatch(int argc, char* argv[])
{
  // Parse for help argument
  struct option options[] = {{"help", no_argument, nullptr, 'h'}, {0, 0, 0, 0}};
  int c;
  int indexptr;
  while((c = getopt_long(argc, argv, "h", options, &indexptr)) != -1)
    switch(c)
    {
    case 'h':
      usageMatch();
      return EXIT_SUCCESS;
    case '?':
      usageMatch();
      return EXIT_FAILURE;
    }

  if(argc == 2)
  {
    App::MatchAgentPlayer(argv[1]).run();
    return EXIT_SUCCESS;
  }
  if(argc == 3)
  {
    App::MatchAgents(argv[1], argv[2]).run();
    return EXIT_SUCCESS;
  }

  usageMatch();
  return EXIT_FAILURE;
}

void usage()
{
  std::clog << "Usage: chinesechess COMMANDS\n";
  std::clog << "\n";

  std::clog << "Commands:\n";
  std::clog << "  generate-populations    Generate populations\n";
  std::clog << "  match                   Play a match between 2 agents or with an agent.\n";
  std::clog << "\n";

  std::clog << "Options:\n";
  std::clog << "  -h,--help\n";
  std::clog << "    print this help message\n";
}

int main(int argc, char* argv[])
{
  // Detect subcommand
  if(argc >= 2)
  {
    if(strcmp("generate-populations", argv[1]) == 0)
      return mainGeneratePopulations(argc-1, argv+1);

    if(strcmp("match", argv[1]) == 0)
      return mainMatch(argc-1, argv+1);
  }

  // Parse for help argument
  struct option options[] = {{"help", no_argument, nullptr, 'h'}, {0, 0, 0, 0}};
  int c;
  int indexptr;
  while((c = getopt_long(argc, argv, "h", options, &indexptr)) != -1)
    switch(c)
    {
    case 'h':
      usage();
      return EXIT_SUCCESS;
    case '?':
      usage();
      return EXIT_FAILURE;
    }

  usage();
  return EXIT_FAILURE;
}
