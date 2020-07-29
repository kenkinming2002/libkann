#include "App.hpp"
#include "Config.hpp"

#include "Grid.hpp"

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <random>
#include <chrono>

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
    std::clog << "No seed provided, generating random seed from ";

    std::random_device rd;
    if(rd.entropy() != 0.0)
    {
      std::clog << "std::random_device...\n";
      seed = rd();
    }
    else
    {
      std::clog << "current time...\n";
      seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }

    std::clog << "Seed is " << seed << '\n';
  }

  App(seed).run();
}
