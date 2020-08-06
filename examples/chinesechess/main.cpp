#include "App.hpp"

#include <filesystem>
#include <sstream>
#include <iostream>
#include <random>
#include <chrono>
#include <optional>

[[gnu::noreturn]] void usage()
{
  std::clog << "Usage: chinesechess OUTPUT_DIRECTORY [SEED]\n";
  std::exit(EXIT_FAILURE);
}

App::seed_type generateSeed()
{
  std::random_device rd;
  if(rd.entropy() != 0.0)
  {
    std::clog << "No seed provided, generating random seed from std::random_device...\n";
    return rd();
  }
  else
  {
    std::clog << "No seed provided, generating random seed from current time...\n";
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }
}

App::seed_type toSeed(const char* str)
{
  App::seed_type seed;

  std::stringstream ss;
  ss << str;
  ss >> seed;
  if(ss.fail() || !ss.eof())
    usage();

  return seed;
}

int main(int argc, const char* argv[])
{
  if(argc < 2 || argc > 3)
    usage();

  auto outputDirectory = std::filesystem::path(argv[1]);
  auto seed = argc == 3 ? toSeed(argv[2]) : generateSeed();

  std::clog << "Running with seed " << seed << '\n';

  App(std::move(outputDirectory), seed).run();
}
