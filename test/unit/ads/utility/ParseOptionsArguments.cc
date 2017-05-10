// -*- C++ -*-

#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <string>

#include <cassert>

using namespace stlib;

int
main(int argc, char* argv[])
{
  {
    ads::ParseOptionsArguments parser(argc, argv);

    assert(parser.getOptionPrefix() == '-');
    assert(parser.getNumberOfArguments() == 0);
    assert(parser.areArgumentsEmpty());

    assert(parser.getProgramName().length() != 0);
    assert(! parser.getOption('a'));
    int value;
    assert(! parser.getOption('a', &value));
  }
  {
    char a0[] = "programName";
    char a1[] = "-a";
    char a2[] = "-alpha";
    char a3[] = "-b=1";
    char a4[] = "-beta=3.45";
    char a5[] = "d";
    char a6[] = "delta";
    char* argVector[] = {a0, a1, a2, a3, a4, a5, a6};
    const int argCount = sizeof(argVector) / sizeof(char*);

    ads::ParseOptionsArguments parser;
    parser.parse(argCount, argVector);

    std::cout << "Options:\n";
    parser.printOptions(std::cout);
    std::cout << "Arguments:\n";
    parser.printArguments(std::cout);

    assert(parser.getOptionPrefix() == '-');

    assert(parser.getNumberOfOptions() == 4);
    assert(! parser.areOptionsEmpty());

    assert(parser.getProgramName() == std::string("programName"));
    assert(parser.getOption('a'));
    assert(! parser.getOption('a'));
    assert(parser.getOption("alpha"));
    assert(! parser.getOption("alpha"));
    assert(! parser.getOption('z'));
    assert(! parser.getOption("zeta"));
    {
      int value;
      assert(parser.getOption('b', &value));
      assert(value == 1);
      assert(! parser.getOption('b', &value));
      assert(! parser.getOption('z', &value));
    }
    {
      double value;
      assert(parser.getOption("beta", &value));
      assert(value == 3.45);
      assert(! parser.getOption("beta", &value));
      assert(! parser.getOption("zeta", &value));
    }

    assert(parser.getNumberOfOptions() == 0);
    assert(parser.areOptionsEmpty());

    assert(parser.getNumberOfArguments() == 2);
    assert(parser.getArgument() == std::string("d"));
    assert(parser.getArgument() == std::string("delta"));
    assert(parser.areArgumentsEmpty());

    std::cout << "Empty options:\n";
    parser.printOptions(std::cout);
    std::cout << "Empty arguments:\n";
    parser.printArguments(std::cout);

    parser.setOptionPrefix('/');
    assert(parser.getOptionPrefix() == '/');
  }

  return 0;
}
