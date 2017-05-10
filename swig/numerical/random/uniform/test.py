from numerical import DiscreteUniformGeneratorMt19937 as Uniform

print 'numerical/random/uniform'

x = Uniform()
for i in range(10):
    print x()

x.seed(0)
a = x()
x.seed(0)
assert a == x()
