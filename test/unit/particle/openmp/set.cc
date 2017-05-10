// -*- C++ -*-

#include "stlib/particle/set.h"

#include <limits>

using namespace stlib;

void
test()
{
  typedef particle::IntegerTypes::Code Code;

  particle::ParticleSet particles;
  assert(particles.empty());
  particles.pack();
  assert(particles.empty());

  particles.append(2, 0);
  assert(particles.empty());

  particles.append(0, 2);
  particles.append(2, 3);
  particles.pack();
  assert(particles.isValid());
  assert(particles.size() == 1);
  assert(particles[0].first == 0);
  assert(particles[0].extent == 5);

  particles.clear();
  particles.append(2, 3);
  particles.append(0, 2);
  particles.pack();
  assert(particles.isValid());
  assert(particles.size() == 1);
  assert(particles[0].first == 0);
  assert(particles[0].extent == 5);

  particles.clear();
  particles.append(3, 3);
  particles.append(0, 2);
  particles.pack();
  assert(particles.isValid());
  assert(particles.size() == 2);
  assert(particles[0].first == 0);
  assert(particles[0].extent == 2);
  assert(particles[1].first == 3);
  assert(particles[1].extent == 3);

  //
  // Append using codes.
  //

  std::vector<Code> codes;
  codes.push_back(1); // 0
  codes.push_back(2); // 1
  codes.push_back(2); // 2
  codes.push_back(3); // 3
  codes.push_back(3); // 4
  codes.push_back(3); // 5
  codes.push_back(std::numeric_limits<Code>::max());

  particles.clear();
  particles.append(0, codes);
  particles.pack();
  assert(particles.isValid());
  assert(particles.size() == 1);
  assert(particles[0].first == 0);
  assert(particles[0].extent == 1);

  particles.append(3, codes);
  particles.pack();
  assert(particles.isValid());
  assert(particles.size() == 2);
  assert(particles[0].first == 0);
  assert(particles[0].extent == 1);
  assert(particles[1].first == 3);
  assert(particles[1].extent == 3);

  particles.append(1, codes);
  particles.pack();
  assert(particles.isValid());
  assert(particles.size() == 1);
  assert(particles[0].first == 0);
  assert(particles[0].extent == 6);
}

int
main()
{
  test();

  return 0;
}
