// -*- C++ -*-

#include "stlib/stochastic/Reaction.h"
#include "stlib/numerical/equality.h"

#include <cmath>

using namespace stlib;

int
main()
{
  typedef stochastic::Reaction<true> ReactionDiscrete;
  typedef stochastic::Reaction<false> ReactionContinuous;
  typedef stochastic::Reaction<true>::SparseVectorSizeType SparseVectorSizeType;
  typedef container::SparseVector<double> StateChangeVector;

  using stochastic::computeDifferenceOfHarmonicNumbers;
  using numerical::areEqual;

  // computeDifferenceOfHarmonicNumbers().
  {
    // d/dx C(x, 0) = d/dx 1 = 0
    assert(computeDifferenceOfHarmonicNumbers(0., 0) == 0);
    assert(computeDifferenceOfHarmonicNumbers(1., 0) == 0);
    // d/dx C(x, 1) = d/dx x = 1
    // H_2 - H_1
    assert(computeDifferenceOfHarmonicNumbers(2., 1) == 0.5);
    assert(computeDifferenceOfHarmonicNumbers(1., 1) == 1.);
    // d/dx C(x, 2) = d/dx x(x-1)/2 = x - 1/2
    // H_3 - H_1
    assert(areEqual(computeDifferenceOfHarmonicNumbers(3., 2), 1. / 3 + 1. / 2));
    // H_2 - H_0
    assert(computeDifferenceOfHarmonicNumbers(2., 2) == 1.5);
  }

  //-------------------------------------------------------------------------
  // Empty reaction.
  {
    // Default constructor.
    ReactionDiscrete x;
    assert(x.getReactants() == SparseVectorSizeType());
    assert(x.getProducts() == SparseVectorSizeType());
    assert(x.getDependencies().empty());
    assert(x.getScaledRateConstant() == 0);
    assert(! stochastic::isValid(x, 0));

    // Copy constructor.
    {
      ReactionDiscrete y(x);
      assert(y == x);
    }
    // Assignment operator.
    {
      ReactionDiscrete y;
      y = x;
      assert(y == x);
    }
  }

  //-------------------------------------------------------------------------
  // s0 ->
  {
    std::cout << "s0 ->\n";
    std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
    std::vector<std::size_t> dependencies(1, 0);
    reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
    const SparseVectorSizeType Reactants(reactants), Products(products);
    const double RateConstant = 2.0;

    // Construct from vectors.
    {
      ReactionDiscrete x(reactants, products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 2));
      // Copy constructor.
      {
        ReactionDiscrete y(x);
        assert(y == x);
      }
      // Assignment operator.
      {
        ReactionDiscrete y;
        y = x;
        assert(y == x);
      }
      // Propensity function.
      {
        std::vector<double> populations;
        populations.push_back(5);
        assert(x.computePropensityFunction(populations) ==
               RateConstant * populations[0]);
        container::SparseVector<double> derivatives;
        x.computePropensityFunctionDerivatives(populations, &derivatives);
        assert(derivatives.size() == 1);
        assert(std::abs(derivatives[0] - RateConstant) <
               10. * std::numeric_limits<double>::epsilon());
        ReactionContinuous y(x);
        assert(y.computePropensityFunction(populations) ==
               RateConstant * populations[0]);

        populations[0] = 0;
        x.computePropensityFunctionDerivatives(populations, &derivatives);
        assert(derivatives.empty());

        {
          // a = 2 x
          // dx/dt = -a = -2 x
          // da/dt = 2(-2 x) = -4 x
          std::vector<double> dxdt(populations.size());
          dxdt[0] = -2 * populations[0];
          assert(x.timeDerivative(populations, dxdt) ==
                 -4 * populations[0]);
        }
      }
      // State change.
      {
        StateChangeVector stateChange = Products - Reactants;
        StateChangeVector sc;
        x.computeStateChangeVector(&sc);
        assert(sc == stateChange);
        std::cout << sc << "\n";
      }
    }
    // Rebuild from vectors.
    {
      ReactionDiscrete x;
      x.rebuild(reactants, products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
    // Construct from sparse arrays.
    {
      ReactionDiscrete x(Reactants, Products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
    // Rebuild from sparse arrays.
    {
      ReactionDiscrete x;
      x.rebuild(Reactants, Products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
    // Use manipulators.
    {
      ReactionDiscrete x;
      x.setReactants(Reactants);
      x.setProducts(Products);
      x.setDependencies(dependencies);
      x.setRateConstant(RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
  }

  //-------------------------------------------------------------------------
  //  -> s0
  {
    std::cout << " -> s0\n";
    std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
    std::vector<std::size_t> dependencies;
    products.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
    const SparseVectorSizeType Reactants(reactants), Products(products);
    const double RateConstant = 2.0;

    // Construct from vectors.
    {
      ReactionDiscrete x(reactants, products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 2));
      // Copy constructor.
      {
        ReactionDiscrete y(x);
        assert(y == x);
      }
      // Assignment operator.
      {
        ReactionDiscrete y;
        y = x;
        assert(y == x);
      }
      // Propensity function.
      {
        std::vector<double> populations;
        populations.push_back(5);
        assert(x.computePropensityFunction(populations) ==
               RateConstant);
        container::SparseVector<double> derivatives;
        x.computePropensityFunctionDerivatives(populations, &derivatives);
        assert(derivatives.size() == 0);
        ReactionContinuous y(x);
        assert(y.computePropensityFunction(populations) ==
               RateConstant);

        populations[0] = 0;
        x.computePropensityFunctionDerivatives(populations, &derivatives);
        assert(derivatives.size() == 0);
        {
          // a = 2
          // dx/dt = a = 2
          // da/dt = 0(2) = 0
          std::vector<double> dxdt(populations.size());
          dxdt[0] = 2;
          assert(x.timeDerivative(populations, dxdt) == 0);
        }
      }
      // State change.
      {
        StateChangeVector stateChange = Products - Reactants;
        StateChangeVector sc;
        x.computeStateChangeVector(&sc);
        assert(sc == stateChange);
        std::cout << sc << "\n";
      }
    }
    // Rebuild from vectors.
    {
      ReactionDiscrete x;
      x.rebuild(reactants, products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
    // Construct from sparse arrays.
    {
      ReactionDiscrete x(Reactants, Products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
    // Rebuild from sparse arrays.
    {
      ReactionDiscrete x;
      x.rebuild(Reactants, Products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
    // Use manipulators.
    {
      ReactionDiscrete x;
      x.setReactants(Reactants);
      x.setProducts(Products);
      x.setDependencies(dependencies);
      x.setRateConstant(RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
  }

  //-------------------------------------------------------------------------
  // 2 s0 ->
  {
    std::cout << "2 s0 ->\n";
    std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
    std::vector<std::size_t> dependencies(1, 0);
    reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 2));
    const SparseVectorSizeType Reactants(reactants), Products(products);
    const double RateConstant = 2.0;

    // Construct from iterators.
    {
      ReactionDiscrete x(reactants, products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 2));
      // Copy constructor.
      {
        ReactionDiscrete y(x);
        assert(y == x);
      }
      // Assignment operator.
      {
        ReactionDiscrete y;
        y = x;
        assert(y == x);
      }
      // Propensity function.
      {
        std::vector<double> populations;
        populations.push_back(5);
        assert(x.computePropensityFunction(populations) ==
               0.5 * RateConstant * populations[0] * (populations[0] - 1));
        container::SparseVector<double> derivatives;
        x.computePropensityFunctionDerivatives(populations, &derivatives);
        assert(derivatives.size() == 1);
        assert(std::abs(derivatives[0] -
                        0.5 * RateConstant * (2 * populations[0] - 1)) <
               10. * std::numeric_limits<double>::epsilon());
        ReactionContinuous y(x);
        assert(y.computePropensityFunction(populations) ==
               0.5 * RateConstant * populations[0] * populations[0]);

        populations[0] = 0;
        x.computePropensityFunctionDerivatives(populations, &derivatives);
        assert(derivatives.empty());

        {
          // a = 2 x (x - 1) / 2 = x (x - 1)
          // dx/dt = (-2) a = -2 x (x - 1)
          // da/dt = (2 x - 1)(-2 x (x - 1)) =
          std::vector<double> dxdt(populations.size());
          const double p = populations[0];
          dxdt[0] = -2 * p * (p - 1);
          assert(x.timeDerivative(populations, dxdt) ==
                 (2 * p - 1) * dxdt[0]);
        }
      }
      // State change.
      {
        StateChangeVector stateChange = Products - Reactants;
        StateChangeVector sc;
        x.computeStateChangeVector(&sc);
        assert(sc == stateChange);
        std::cout << sc << "\n";
      }
    }
    // Rebuild from iterators.
    {
      ReactionDiscrete x;
      x.rebuild(reactants, products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
    // Construct from sparse arrays.
    {
      ReactionDiscrete x(Reactants, Products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
    // Rebuild from sparse arrays.
    {
      ReactionDiscrete x;
      x.rebuild(Reactants, Products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
    // Use manipulators.
    {
      ReactionDiscrete x;
      x.setReactants(Reactants);
      x.setProducts(Products);
      x.setDependencies(dependencies);
      x.setRateConstant(RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 1));
    }
  }

  //-------------------------------------------------------------------------
  // s0 -> s1
  {
    std::cout << "s0 -> s1\n";
    std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
    reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
    products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
    const SparseVectorSizeType Reactants(reactants), Products(products);
    std::vector<std::size_t> dependencies(1, 0);
    const double RateConstant = 2.0;

    // Construct from iterators.
    {
      ReactionDiscrete x(reactants, products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 2));
      // Copy constructor.
      {
        ReactionDiscrete y(x);
        assert(y == x);
      }
      // Assignment operator.
      {
        ReactionDiscrete y;
        y = x;
        assert(y == x);
      }
      // Propensity function.
      {
        std::vector<double> populations;
        populations.push_back(5);
        populations.push_back(7);
        assert(x.computePropensityFunction(populations) ==
               RateConstant * populations[0]);
        container::SparseVector<double> derivatives;
        x.computePropensityFunctionDerivatives(populations, &derivatives);
        assert(derivatives.size() == 1);
        assert(std::abs(derivatives[0] - RateConstant) <
               10. * std::numeric_limits<double>::epsilon());
        ReactionContinuous y(x);
        assert(y.computePropensityFunction(populations) ==
               RateConstant * populations[0]);

        populations[0] = 0;
        x.computePropensityFunctionDerivatives(populations, &derivatives);
        assert(derivatives.empty());

        {
          // s0 -> s1
          // a = 2 s0
          // ds0/dt = (-1) a = -2 s0
          // ds1/dt = (1) a = 2 s0
          // da/dt = pa/ps0 ds0/dt = 2 (-2 s0) = -4 s0
          std::vector<double> dxdt(populations.size());
          dxdt[0] = -2 * populations[0];
          dxdt[1] = 2 * populations[0];
          assert(x.timeDerivative(populations, dxdt) ==
                 -4 * populations[0]);
        }
      }
      // State change.
      {
        StateChangeVector stateChange = Products - Reactants;
        StateChangeVector sc;
        x.computeStateChangeVector(&sc);
        assert(sc == stateChange);
        std::cout << sc << "\n";
      }
    }
    // Rebuild from iterators.
    {
      ReactionDiscrete x;
      x.rebuild(reactants, products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 2));
    }
    // Construct from sparse arrays.
    {
      ReactionDiscrete x(Reactants, Products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 2));
    }
    // Rebuild from sparse arrays.
    {
      ReactionDiscrete x;
      x.rebuild(Reactants, Products, dependencies, RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 2));
    }
    // Use manipulators.
    {
      ReactionDiscrete x;
      x.setReactants(Reactants);
      x.setProducts(Products);
      x.setDependencies(dependencies);
      x.setRateConstant(RateConstant);
      assert(x.getReactants() == Reactants);
      assert(x.getProducts() == Products);
      assert(x.getDependencies() == dependencies);
      assert(x.computeRateConstant() == RateConstant);
      assert(stochastic::isValid(x, 2));
    }
  }

  return 0;
}
