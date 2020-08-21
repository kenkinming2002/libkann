#include "Match.hpp"

#include "Match/AgentPlayer.hpp"
#include "Match/Agents.hpp"

#include <getopt.h>

namespace App
{
  void Match::usage()
  {
    std::clog << "Usage: chinesechess match AGENT1 [AGENT2]\n";
    std::clog << "\n";

    std::clog << "Match between the 2 agents if AGENT2 is specified. Otherwise, the user will play against AGENT1.\n";
    std::clog << "\n";

    std::clog << "Options:\n";
    std::clog << "  -h,--help\n";
    std::clog << "    print this help message\n";
  }

  int Match::main(int argc, char* argv[])
  {
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

    if(argc == 2)
    {
      AgentPlayer(argv[1]).run();
      return EXIT_SUCCESS;
    }
    if(argc == 3)
    {
      Agents(argv[1], argv[2]).run();
      return EXIT_SUCCESS;
    }

    usage();
    return EXIT_FAILURE;
  }
}
