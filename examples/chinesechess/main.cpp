#include "Population.hpp"
#include "Game.hpp"

#include "HumanAgent.hpp"
#include "AIAgent.hpp"
#include "MinimaxAgent.hpp"

#include "cereal/archives/binary.hpp"
#include "utilities/lexical_cast.hpp"

#include <libkann/utilities/random.hpp>

#include <iostream>
#include <getopt.h>
#include <random>

void generate_populations_usage()
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

int generate_populations(int argc, char* argv[])
{
  typedef std::default_random_engine::result_type seed_type;

  // Program options
  const char* outputDirectory = nullptr;
  std::optional<seed_type> populationSeed;
  std::optional<size_t> populationSize;
  std::optional<size_t> selectionIterationsCount;
  {
    // Parsing
    enum OptionVal : int
    {
      POPULATION_SEED = 0,
      POPULATION_SIZE = 1,
      SELECTION_ITERATIONS_COUNT = 2
    };

    struct option options[] =
    {
      {"output"                    , required_argument, nullptr, 'o'},
      {"population-seed"           , required_argument, nullptr, POPULATION_SEED},
      {"population-size"           , required_argument, nullptr, POPULATION_SIZE},
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
      case SELECTION_ITERATIONS_COUNT:
        selectionIterationsCount = lexical_cast<size_t>(optarg);
        break;
      case 'h':
        generate_populations_usage();
        return EXIT_SUCCESS;
      case '?':
        generate_populations_usage();
        return EXIT_FAILURE;
    }

    // Default arguments
    if(optind < argc)
    {
      std::clog << "error: too many arguments\n";
      generate_populations_usage();
      return EXIT_FAILURE;
    }

    if(!outputDirectory)
    {
      std::clog << "error: output directory not specified\n";
      generate_populations_usage();
      return EXIT_FAILURE;
    }
    if(!populationSeed) populationSeed = random<seed_type>();
    if(!populationSize) populationSize = 5000;
    if(!selectionIterationsCount) selectionIterationsCount = 10;

    // DEBUG
    std::clog << "DEBUG: outputDirectory=" << (outputDirectory ? outputDirectory : "") << '\n';
    std::clog << "DEBUG: populationSeed=" << *populationSeed << '\n';
    std::clog << "DEBUG: populationSize=" << *populationSize << '\n';
    std::clog << "DEBUG: selectionIterationsCount=" << *selectionIterationsCount << '\n';
  }

  std::default_random_engine engine(*populationSeed);
  Population population(*populationSize);

  for(size_t i=0;;++i)
  {
    std::cout << "Writing population " << i << "..." << std::flush;
    population.write(std::filesystem::path(outputDirectory) / ("population" + std::to_string(i)));
    std::cout << "Done\n";

    std::clog << "Generating population " << i+1 << "...\n";
    population.select(*selectionIterationsCount, engine, 0.5);
    std::clog << "Population " << i << " generated\n";
  }

  return EXIT_SUCCESS;
}

void match_usage()
{
  std::clog << "Usage: chinesechess match AGENT1 AGENT2\n";
  std::clog << "\n";

  std::clog << "Match between the 2 agents\n";
  std::clog << "\n";

  std::clog << "Options:\n";
  std::clog << "  -h,--help\n";
  std::clog << "    print this help message\n";
}

std::unique_ptr<Agent> match_open_agent(char* uri)
{
  // C dark magic to avoid memory allocation
  char* colon = strchr(uri, ':');
  if(!colon)
  {
    std::cerr << "Error: Uri " << uri << " missing : delimiter\n";
    return nullptr;
  }

  *colon = '\0';

  std::string_view scheme = uri;
  std::string_view name = colon+1;

  if(scheme == "human")
  {
    std::clog << "LOG: Creating human agent with name=" << name << '\n';
    return std::make_unique<HumanAgent>();
  }
  else if(scheme == "ai")
  {
    AIAgent agent;

    std::ifstream file(std::string(name.data(), name.length()));
    cereal::BinaryInputArchive archive(file);
    archive(agent);

    return std::make_unique<AIAgent>(std::move(agent));
  }
  else if(scheme == "minimax")
  {
    std::clog << "LOG: Creating minimax agent with name=" << name << '\n';
    return std::make_unique<MinimaxAgent>();
  }
  else
  {
    std::cerr << "Error: Unknown scheme " << scheme << "\n";
    return nullptr;
  }
}

int match(int argc, char* argv[])
{
  // Parse for help argument
  struct option options[] = {{"help", no_argument, nullptr, 'h'}, {0, 0, 0, 0}};
  int c;
  int indexptr;
  while((c = getopt_long(argc, argv, "h", options, &indexptr)) != -1)
    switch(c)
    {
    case 'h':
      match_usage();
      return EXIT_SUCCESS;
    case '?':
      match_usage();
      return EXIT_FAILURE;
    }

  if(argc != 3)
  {
    match_usage();
    return EXIT_FAILURE;
  }

  auto agent1 = match_open_agent(argv[1]);
  if(!agent1)
    return EXIT_FAILURE;

  auto agent2 = match_open_agent(argv[2]);
  if(!agent2)
    return EXIT_FAILURE;

  auto result = game(*agent1, *agent2);
  return EXIT_SUCCESS;
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
      return generate_populations(argc-1, argv+1);

    if(strcmp("match", argv[1]) == 0)
      return match(argc-1, argv+1);
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
