#include "GeneratePopulations.hpp"

#include "../utilities/lexical_cast.hpp"

#include <libkann/utilities/random.hpp>

#include <iostream>
#include <getopt.h>

namespace App
{
  void GeneratePopulations::usage()
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

  int GeneratePopulations::main(int argc, char* argv[])
  {
    // Program options
    const char* outputDirectory = nullptr;
    std::optional<seed_type> populationSeed;
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
        populationSeed = lexical_cast<seed_type>(optarg);
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

  if(!outputDirectory) 
  { 
    std::clog << "error: output directory not specified\n";
    usage(); 
    return EXIT_FAILURE; 
  }
  if(!populationSeed) populationSeed = random<seed_type>();
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

  GeneratePopulations::GeneratePopulations(const char* outputDirectory, seed_type populationSeed, size_t populationSize, const std::vector<size_t>& agentsHiddenLayers, size_t selectionIterationsCount) 
    : m_outputDirectory(outputDirectory), m_population(populationSeed, populationSize, agentsHiddenLayers), m_selectionIterations(selectionIterationsCount)
  {
    std::clog << "Running with seed " << populationSeed << '\n';
  }

  void GeneratePopulations::run()
  {
    // Initial
    auto write = [this](size_t i) {
      std::cout << "Writing population " << i << "..." << std::flush;
      m_population.writeTo(m_outputDirectory / ("population" + std::to_string(i)));
      std::cout << "Done\n";
    };

    write(0);
    for(size_t i=1;;++i)
    {
      std::clog << "Generating population " << i << "...\n";
      m_population.select(m_selectionIterations);
      write(i);
      std::clog << "Population " << i << " generated\n";
    }
  }
}
