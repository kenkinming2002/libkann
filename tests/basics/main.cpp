#include <libkann/version.hpp>
#include <libkann/kann.hpp>

#include "App.hpp"
#include "Config.hpp"

#include <iostream>
#include <cstdlib>

[[gnu::noreturn]] void usage()
{
  std::clog << "Usage: driver CONFIG_FILE\n";
  std::exit(EXIT_FAILURE);
}

int main (int argc, const char* argv[])
{
  if(argc != 2)
    usage();

  CONFIG = Config(argv[1]);

  App().run();
}
