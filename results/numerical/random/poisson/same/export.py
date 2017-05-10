import os

for name in os.listdir("."):
    suffix = os.path.splitext(name)[-1]
    if suffix == '.jpg' or suffix == '.pdf':
	os.system('cp ' + name + ' ../../../../../doxygen/numerical/graphics/random/poisson/same/same' + name)
