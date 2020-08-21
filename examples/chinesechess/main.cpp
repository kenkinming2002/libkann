#include "App.hpp"

#include <iostream>
#include <getopt.h>

void usage()
{
  std::clog << "Usage: chinesechess COMMANDS\n";
  std::clog << "\n";

  std::clog << "Commands:\n";
  std::clog << "  generate-populations    Generate populations\n";
  std::clog << "  match                   Play a match between 2 agents or with an agent.\n";
  std::clog << "  hell                    Play agaist an agent whose sole purpose is to beat you and will learn from you.\n";
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
      return App::GeneratePopulations::main(argc-1, argv+1);

    if(strcmp("match", argv[1]) == 0)
      return App::Match::main(argc-1, argv+1);

    if(strcmp("hell", argv[1]) == 0)
      return App::Hell::main(argc-1, argv+1);
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
