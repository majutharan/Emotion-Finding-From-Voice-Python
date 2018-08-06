import numpy

array = [1, 2, 3, 4, 5]
print(array)
test = []
for a in array:
    print(a)
    test.append(a)
    numpy.savetxt("test.csv", test, fmt="%s")


print(test)