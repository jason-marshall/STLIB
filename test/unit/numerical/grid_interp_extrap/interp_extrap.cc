// -*- C++ -*-

#include "stlib/numerical/grid_interp_extrap/interp_extrap.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

int
main()
{
  //
  // 1-D, 1 field.
  //
  {
    double value;
    bool isValueKnown;
    std::array<double, 4> fields = {{0.0, 1.0, 2.0, 3.0}};
    std::array<bool, 4> isKnown;
    double position;
    std::ptrdiff_t lowerIndex;

    position = 0.5;
    lowerIndex = -1;
    std::cout << "Interpolation for " << fields << '\n'
              << "lowerIndex = " << lowerIndex
              << ", position = " << position << '\n';
    std::size_t i;
    for (std::size_t n = 0; n != 16; ++n) {
      i = n;
      isKnown[0] = (i / 8);
      i %= 8;
      isKnown[1] = (i / 4);
      i %= 4;
      isKnown[2] = (i / 2);
      i %= 2;
      isKnown[3] = i;
      numerical::intExt(&value, &isValueKnown, fields, isKnown,
                        position, lowerIndex);
      std::cout << isKnown << " -> "
                << (isValueKnown ? value : 0.0) << " "
                << isValueKnown
                << '\n';
    }


    isKnown = std::array<bool, 4>{
      {
        false, false, false, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(! isValueKnown);

    isKnown = std::array<bool, 4>{
      {
        false, false, false, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 3.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, false, true, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, false, true, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, true, false, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, true, false, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, true, true, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, true, true, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, false, false, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 0.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, false, false, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           (std::abs(value - 0.0) <
            10.0 * std::numeric_limits<double>::epsilon() ||
            std::abs(value - 3.0) <
            10.0 * std::numeric_limits<double>::epsilon()));

    isKnown = std::array<bool, 4>{
      {
        true, false, true, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, false, true, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, true, false, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, true, false, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, true, true, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, true, true, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon());

  }



  //
  // 1-D, 1 field, user interface.
  //
  {
    const std::size_t N = 1, M = 1;
    std::vector<std::array<double, M> > values(1);
    values[0][0] = std::numeric_limits<double>::max();
    std::vector<std::array<double, N> > positions(1);
    std::array<double, M> defaultValues = {{ -1}};
    std::array<std::size_t, N> extents = {{4}};
    std::array<double, N> lowerCorner = {{0}};
    std::array<double, N> upperCorner = {{1}};
    double distance[4];
    const double fields[4] = {0.0, 1.0, 2.0, 3.0};

    //
    // No interpolation/extrapolation if there are not 4 surrounding grid
    // points.
    //
    positions[0][0] = 0.25;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance,
                                fields);
    assert(values[0][0] == std::numeric_limits<double>::max());

    positions[0][0] = 0.75;
    numerical::gridInterpExtrap(&values, positions,
                                defaultValues, extents,
                                lowerCorner, upperCorner, distance, fields);
    assert(values[0][0] == std::numeric_limits<double>::max());


    positions[0][0] = 0.5;

    // I don't need the output.  There are assertion below.
#if 0
    std::cout << "\n1-D interpolation.\n";
    std::size_t i;
    for (std::size_t n = 0; n != 16; ++n) {
      i = n;
      distance[0] = i / 8 ? 1 : -1;
      i %= 8;
      distance[1] = i / 4 ? 1 : -1;
      i %= 4;
      distance[2] = i / 2 ? 1 : -1;
      i %= 2;
      distance[3] = i ? 1 : -1;
      numerical::gridInterpExtrap(&values, positions,
                                  defaultValues, extents,
                                  lowerCorner, upperCorner, distance, fields);
      std::cout << distance[0] << " " << distance[1] << " "
                << distance[2] << " " << distance[3] << " -> "
                << values[0] << '\n';
    }
#endif

    distance[0] = -1;
    distance[1] = -1;
    distance[2] = -1;
    distance[3] = -1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(values[0] == defaultValues);

    distance[0] = -1;
    distance[1] = -1;
    distance[2] = -1;
    distance[3] = 1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 3.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = -1;
    distance[1] = -1;
    distance[2] = 1;
    distance[3] = -1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = -1;
    distance[1] = -1;
    distance[2] = 1;
    distance[3] = 1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = -1;
    distance[1] = 1;
    distance[2] = -1;
    distance[3] = -1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = -1;
    distance[1] = 1;
    distance[2] = -1;
    distance[3] = 1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = -1;
    distance[1] = 1;
    distance[2] = 1;
    distance[3] = -1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = -1;
    distance[1] = 1;
    distance[2] = 1;
    distance[3] = 1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = 1;
    distance[1] = -1;
    distance[2] = -1;
    distance[3] = -1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 0.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = 1;
    distance[1] = -1;
    distance[2] = -1;
    distance[3] = 1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 0.0) <
           10.0 * std::numeric_limits<double>::epsilon() ||
           std::abs(values[0][0] - 3.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = 1;
    distance[1] = -1;
    distance[2] = 1;
    distance[3] = -1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = 1;
    distance[1] = -1;
    distance[2] = 1;
    distance[3] = 1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = 1;
    distance[1] = 1;
    distance[2] = -1;
    distance[3] = -1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = 1;
    distance[1] = 1;
    distance[2] = -1;
    distance[3] = 1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = 1;
    distance[1] = 1;
    distance[2] = 1;
    distance[3] = -1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon());

    distance[0] = 1;
    distance[1] = 1;
    distance[2] = 1;
    distance[3] = 1;
    numerical::gridInterpExtrap(&values, positions, defaultValues,
                                extents, lowerCorner, upperCorner, distance, fields);
    assert(std::abs(values[0][0] - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon());
  }

  //
  // 1-D, 1 field, pointer interface.
  //
  {
    const std::size_t Size = 1, N = 1, M = 1;
    double values[Size * M] = {std::numeric_limits<double>::max()};
    double positions[Size * N] = {};
    double defaultValues[M] = { -1};
    int extents[N] = {4};
    double lowerCorner[N] = {0};
    double upperCorner[N] = {1};
    const double distance[4] = {};
    const double fields[4] = {0.0, 1.0, 2.0, 3.0};

    positions[0] = 0.25;
    // Pointer interface.
    numerical::gridInterpExtrap<N, M>(Size, values, positions, defaultValues,
                                      extents, lowerCorner, upperCorner,
                                      distance, fields);
    assert(values[0] == std::numeric_limits<double>::max());
  }


  //
  // 1-D, 2 fields.
  //
  {
    std::array<double, 2> value;
    bool isValueKnown;
    std::array<std::array<double, 2>, 4> fields;
    std::array<bool, 4> isKnown;
    double position;
    std::ptrdiff_t lowerIndex;

    fields[0][0] = 0;
    fields[1][0] = 1;
    fields[2][0] = 2;
    fields[3][0] = 3;

    fields[0][1] = 1;
    fields[1][1] = 2;
    fields[2][1] = 3;
    fields[3][1] = 4;

    position = 0.5;
    lowerIndex = -1;
    std::cout << "Interpolation for " << fields << '\n'
              << "lowerIndex = " << lowerIndex
              << ", position = " << position << '\n';
    std::size_t i;
    for (std::size_t n = 0; n != 16; ++n) {
      i = n;
      isKnown[0] = (i / 8);
      i %= 8;
      isKnown[1] = (i / 4);
      i %= 4;
      isKnown[2] = (i / 2);
      i %= 2;
      isKnown[3] = i;
      numerical::intExt(&value, &isValueKnown, fields, isKnown,
                        position, lowerIndex);
      std::cout << isKnown << " -> "
                << (isValueKnown ? value : std::array<double, 2>{{0., 0.}})
                << " "
                << isValueKnown
                << '\n';
    }


    isKnown = std::array<bool, 4>{
      {
        false, false, false, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown,
                      position, lowerIndex);
    assert(! isValueKnown);

    isKnown = std::array<bool, 4>{
      {
        false, false, false, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 3) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 4) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, false, true, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 2) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 3) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, false, true, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 3.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, true, false, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 1) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 2) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, true, false, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 1) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 2) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, true, true, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 2.5) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        false, true, true, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 2.5) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, false, false, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 0) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 1) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, false, false, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           ((std::abs(value[0] - 0) <
             10.0 * std::numeric_limits<double>::epsilon() &&
             std::abs(value[1] - 1) <
             10.0 * std::numeric_limits<double>::epsilon()) ||
            (std::abs(value[0] - 3) <
             10.0 * std::numeric_limits<double>::epsilon() &&
             std::abs(value[1] - 4) <
             10.0 * std::numeric_limits<double>::epsilon())));

    isKnown = std::array<bool, 4>{
      {
        true, false, true, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 2) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 3) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, false, true, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 3.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, true, false, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, true, false, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 1.0) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 2.0) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, true, true, false
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 2.5) <
           10.0 * std::numeric_limits<double>::epsilon());

    isKnown = std::array<bool, 4>{
      {
        true, true, true, true
      }
    };
    numerical::intExt(&value, &isValueKnown, fields, isKnown, position, lowerIndex);
    assert(isValueKnown &&
           std::abs(value[0] - 1.5) <
           10.0 * std::numeric_limits<double>::epsilon() &&
           std::abs(value[1] - 2.5) <
           10.0 * std::numeric_limits<double>::epsilon());
  }




  //
  // 3-D interpolation, 1 field.
  //
  {
    typedef container::MultiIndexTypes<3>::SizeList SizeList;

    std::array<double, 1> values;
    std::array<double, 3> position = {{1.5, 1.5, 1.5}};
    double fieldsData[4 * 4 * 4];
    const SizeList extents = {{4, 4, 4}};

    container::MultiArrayConstRef<double, 4>
      fields(fieldsData, std::array<std::size_t, 4>{{1, 4, 4, 4}});
    double* ptr = fieldsData;
    for (std::size_t i = 0; i != 4; ++i) {
      for (std::size_t j = 0; j != 4; ++j) {
        for (std::size_t k = 0; k != 4; ++k) {
          *ptr++ = i + j + k;
        }
      }
    }

    double distanceData[4 * 4 * 4];
    container::MultiArrayConstRef<double, 3> distance(distanceData, extents);
    geom::BBox<double, 3> domain = {{{0., 0., 0.}}, {{3., 3., 3.}}};
    geom::RegularGrid<3> grid(extents, domain);
    std::array<double, 1> defaultValues = {{0.0}};

    // The number of tests.
    for (std::size_t n = 0; n != 1000; ++n) {
      // Loop over the distance array.
      for (std::size_t i = 0; i != distance.size(); ++i) {
        // Set the distance to +-1.
        distanceData[i] = 2 * (rand() % 2) - 1;
      }
      numerical::intExt(&values, position, defaultValues, grid, distance, fields);
      assert(0 <= values[0] && values[0] <= 9);
    }
  }



  //
  // 3-D interpolation, 1 field, set of points.
  //
  {
    std::vector<std::array<double, 1> > values(27);
    for (std::size_t i = 0; i != values.size(); ++i) {
      values[i][0] = std::numeric_limits<double>::max();
    }

    std::vector<std::array<double, 3> > positions(27);
    std::vector<std::array<double, 3> >::iterator iter
      = positions.begin();
    for (std::size_t i = 1; i != 4; ++i) {
      for (std::size_t j = 1; j != 4; ++j) {
        for (std::size_t k = 1; k != 4; ++k) {
          *iter++ = std::array<double, 3>{{i / 4.0, j / 4.0, k / 4.0}};
        }
      }
    }

    std::array<double, 1> defaultValues = {{0.0}};

    const std::array<std::size_t, 3> extents = {{4, 4, 4}};
    geom::BBox<double, 3> domain = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    geom::RegularGrid<3> grid(extents, domain);

    double distanceData[4 * 4 * 4];
    container::MultiArrayConstRef<double, 3> distance(distanceData, extents);

    double fieldsData[4 * 4 * 4];
    container::MultiArrayConstRef<double, 4>
      fields(fieldsData, std::array<std::size_t, 4>{{1, 4, 4, 4}});
    double* ptr = fieldsData;
    for (std::size_t i = 0; i != 4; ++i) {
      for (std::size_t j = 0; j != 4; ++j) {
        for (std::size_t k = 0; k != 4; ++k) {
          *ptr++ = i + j + k;
        }
      }
    }

    // The number of tests.
    for (std::size_t n = 0; n != 1000; ++n) {
      // Loop over the distance array.
      for (std::size_t i = 0; i != distance.size(); ++i) {
        // Set the distance to +-1.
        distanceData[i] = 2 * (rand() % 2) - 1;
      }
      numerical::gridInterpExtrap(&values, positions, defaultValues, grid,
                                  distance, fields);
      for (std::size_t j = 0; j != values.size(); ++j) {
        assert((0 <= values[j][0] && values[j][0] <= 9) ||
               values[j][0] == std::numeric_limits<double>::max());
      }
    }
  }


  //
  // 2-D interpolation, 1 field, set of points, user interface.
  //
  {
    const std::size_t N = 2, M = 1;

    std::vector<std::array<double, M> > values(9);
    for (std::size_t i = 0; i != values.size(); ++i) {
      values[i][0] = std::numeric_limits<double>::max();
    }

    std::vector<std::array<double, N> > positions(values.size());
    std::vector<std::array<double, N> >::iterator iter
      = positions.begin();
    for (std::size_t i = 1; i != 4; ++i) {
      for (std::size_t j = 1; j != 4; ++j) {
        *iter++ = std::array<double, N>{{i / 4.0, j / 4.0}};
      }
    }

    std::array<double, M> defaultValues = {{0.0}};

    const std::array<std::size_t, N> extents = {{4, 4}};
    const std::array<double, N> lowerCorner = {{0, 0}};
    const std::array<double, N> upperCorner = {{1, 1}};

    double distance[ 4 * 4 ];

    double fields[4 * 4];
    {
      double* p = fields;
      for (std::size_t i = 0; i != 4; ++i) {
        for (std::size_t j = 0; j != 4; ++j) {
          *p++ = i + j;
        }
      }
    }

    // The number of tests.
    for (std::size_t n = 0; n != 1000; ++n) {
      // Loop over the distance array.
      for (std::size_t i = 0; i != 16; ++i) {
        // Set the distance to +-1.
        distance[i] = 2 * (rand() % 2) - 1;
      }
      numerical::gridInterpExtrap(&values, positions, defaultValues, extents,
                                  lowerCorner, upperCorner, distance, fields);
      for (std::size_t j = 0; j != values.size(); ++j) {
        assert((0 <= values[j][0] && values[j][0] <= 6) ||
               values[j][0] == std::numeric_limits<double>::max());
      }
    }
  }


  //
  // 3-D interpolation, 1 field, set of points, user interface.
  //
  {
    const std::size_t N = 3, M = 1;

    std::vector<std::array<double, M> > values(27);
    for (std::size_t i = 0; i != values.size(); ++i) {
      values[i][0] = std::numeric_limits<double>::max();
    }

    std::vector<std::array<double, N> > positions(values.size());
    std::vector<std::array<double, N> >::iterator iter
      = positions.begin();
    for (std::size_t i = 1; i != 4; ++i) {
      for (std::size_t j = 1; j != 4; ++j) {
        for (std::size_t k = 1; k != 4; ++k) {
          *iter++ = std::array<double, N>{{i / 4.0, j / 4.0, k / 4.0}};
        }
      }
    }

    std::array<double, M> defaultValues = {{0.0}};

    const std::array<std::size_t, N> extents = {{4, 4, 4}};
    const std::array<double, N> lowerCorner = {{0, 0, 0}};
    const std::array<double, N> upperCorner = {{1, 1, 1}};

    double distance[ 4 * 4 * 4 ];

    double fields[4 * 4 * 4];
    {
      double* p = fields;
      for (std::size_t i = 0; i != 4; ++i) {
        for (std::size_t j = 0; j != 4; ++j) {
          for (std::size_t k = 0; k != 4; ++k) {
            *p++ = i + j + k;
          }
        }
      }
    }

    // The number of tests.
    for (std::size_t n = 0; n != 1000; ++n) {
      // Loop over the distance array.
      for (std::size_t i = 0; i != 64; ++i) {
        // Set the distance to +-1.
        distance[i] = 2 * (rand() % 2) - 1;
      }
      numerical::gridInterpExtrap(&values, positions, defaultValues, extents,
                                  lowerCorner, upperCorner, distance, fields);
      for (std::size_t j = 0; j != values.size(); ++j) {
        assert((0 <= values[j][0] && values[j][0] <= 9) ||
               values[j][0] == std::numeric_limits<double>::max());
      }
    }
  }


  //
  // 3-D interpolation, 2 fields, single array, set of points, user interface.
  //
  {
    const std::size_t N = 3, M = 2;

    std::vector<std::array<double, M> > values(27);
    for (std::size_t i = 0; i != values.size(); ++i) {
      for (std::size_t m = 0; m != M; ++m) {
        values[i][m] = std::numeric_limits<double>::max();
      }
    }

    std::vector<std::array<double, N> > positions(values.size());
    std::vector<std::array<double, N> >::iterator iter
      = positions.begin();
    for (std::size_t i = 1; i != 4; ++i) {
      for (std::size_t j = 1; j != 4; ++j) {
        for (std::size_t k = 1; k != 4; ++k) {
          *iter++ = std::array<double, N>{{i / 4.0, j / 4.0, k / 4.0}};
        }
      }
    }

    std::array<double, M> defaultValues = {{0, 1}};

    const std::array<std::size_t, N> extents = {{4, 4, 4}};
    const std::array<double, N> lowerCorner = {{0, 0, 0}};
    const std::array<double, N> upperCorner = {{1, 1, 1}};

    double distance[ 4 * 4 * 4 ];

    double fields[M * 4 * 4 * 4];
    {
      double* p = fields;
      for (std::size_t i = 0; i != 4; ++i) {
        for (std::size_t j = 0; j != 4; ++j) {
          for (std::size_t k = 0; k != 4; ++k) {
            *p++ = i + j + k;
            *p++ = i + j + k + 1;
          }
        }
      }
    }

    // The number of tests.
    for (std::size_t n = 0; n != 1000; ++n) {
      // Loop over the distance array.
      for (std::size_t i = 0; i != 64; ++i) {
        // Set the distance to +-1.
        distance[i] = 2 * (rand() % 2) - 1;
      }
      numerical::gridInterpExtrap(&values, positions, defaultValues, extents,
                                  lowerCorner, upperCorner, distance, fields);
      for (std::size_t j = 0; j != values.size(); ++j) {
        assert((0 <= values[j][0] && values[j][0] <= 9) ||
               values[j][0] == std::numeric_limits<double>::max());
        assert((0 <= values[j][1] && values[j][1] <= 10) ||
               values[j][1] == std::numeric_limits<double>::max());
      }
    }
  }

  return 0;
}
