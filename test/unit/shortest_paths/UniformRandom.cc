// -*- C++ -*-

#include "stlib/shortest_paths/UniformRandom.h"

#include <iostream>

int
main()
{
  std::cout << "Random doubles" << '\n';
  {
    UniformRandom<double> random(1);
    std::cout << "ratio = 1: " << random() << " "
              << random() << " "
              << random() << " "
              << random() << '\n';
  }
  {
    UniformRandom<double> random(2);
    std::cout << "ratio = 2: " << random() << " "
              << random() << " "
              << random() << " "
              << random() << '\n';
  }
  {
    UniformRandom<double> random(0);
    std::cout << "ratio = inf: " << random() << " "
              << random() << " "
              << random() << " "
              << random() << '\n';
  }

  std::cout << "Random integers" << '\n';
  {
    UniformRandom<int> random(1);
    std::cout << "ratio = 1: " << random() << " "
              << random() << " "
              << random() << " "
              << random() << '\n';
  }
  {
    UniformRandom<int> random(2);
    std::cout << "ratio = 2: " << random() << " "
              << random() << " "
              << random() << " "
              << random() << '\n';
  }
  {
    UniformRandom<int> random(0);
    std::cout << "ratio = inf: " << random() << " "
              << random() << " "
              << random() << " "
              << random() << '\n';
  }
  return 0;
}
