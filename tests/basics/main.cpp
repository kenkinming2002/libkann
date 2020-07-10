#include <libkann/version.hpp>
#include <libkann/kann.hpp>

#include "App.hpp"
#include "Config.hpp"

#include "Grid.hpp"

#include <iostream>
#include <cstdlib>
#include <sstream>

[[gnu::noreturn]] void usage()
{
  std::clog << "Usage: driver CONFIG_FILE [SEED]\n";
  std::exit(EXIT_FAILURE);
}

int main (int argc, const char* argv[])
{
  if(argc < 2 || argc > 3)
    usage();

  CONFIG = Config(argv[1]);

  App::seed_type seed;
  if(argc == 3)
  {
    std::stringstream ss;
    ss << argv[2];
    ss >> seed;
    if(ss.fail() || !ss.eof())
      usage();
  }
  else
  {
    seed = std::random_device()();
    std::clog << "No seed provided - using random seed " << seed << '\n';
  }

  App(seed).run();
}
