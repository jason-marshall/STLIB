// -*- C++ -*-

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

using namespace stlib;

namespace
{
//! The program name.
static std::string programName;

//! Exit with an error message.
void
exitOnError()
{
  std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " size\n"
      << "size is the number of records.\n";
  exit(1);
}
}

int
main(int argc, char* argv[])
{
  ads::ParseOptionsArguments parser(argc, argv);

  // Program name.
  programName = parser.getProgramName();

  // Get the number of records.
  if (parser.getNumberOfArguments() != 1) {
    exitOnError();
  }
  std::size_t size = 0;
  parser.getArgument(&size);
  if (size == 0) {
    exitOnError();
  }

  // There should be no more arguments.
  if (! parser.areArgumentsEmpty()) {
    std::cerr << "Error: Un-parsed arguments:\n";
    exitOnError();
  }

  // There should be no options.
  if (! parser.areOptionsEmpty()) {
    std::cerr << "Error: Unmatched options:\n";
    exitOnError();
  }

  std::vector<std::size_t> records(size);
  for (std::size_t n = 0; n != size; ++n) {
    records[n] = n;
  }

  std::map<std::size_t, std::size_t> x;
  std::map<std::size_t, std::size_t>::iterator position = x.begin();
  for (std::size_t n = 0; n != size; ++n) {
    position = x.insert(position, std::make_pair(records[n], records[n]));
  }
  std::random_shuffle(records.begin(), records.end());

  // Time the generator.
  ads::Timer timer;
  std::size_t count = 1000;
  double elapsedTime;
  std::size_t result = 0;
  do {
    count *= 2;
    timer.tic();
    std::size_t c = 0;
    while (c != count) {
      for (std::size_t n = 0; n != size && c != count; ++n) {
        result += x.count(records[n]);
        ++c;
      }
    }
    elapsedTime = timer.toc();
  }
  while (elapsedTime < 1);

  std::cout
      << "Meaningless result = " << result << "\n"
      << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
      << count << " operations in "
      << elapsedTime << " seconds.\n"
      << "Time per operation in nanoseconds:\n";
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(0);
  std::cout << elapsedTime / count * 1e9 << "\n";

  return 0;
}
