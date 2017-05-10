from numerical import DiscreteUniformGeneratorMt19937 as Uniform, ExponentialGeneratorZigguratDefault as Exponential

print 'numerical/random/exponential'

uniform = Uniform()
x = Exponential(uniform)
for i in range(10):
    print x()

x.seed(0)
a = x()
x.seed(0)
assert a == x()
