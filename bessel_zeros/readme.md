To generate the sequence of zeros of a given Bessel function a [C++ Boost](https://www.boost.org/) library is required.

Then the following steps have to be performed:

1. 	cd /bessel_zeros
2.	c++ -I ./boost bessel_zeros.cpp -o bessel_zeros
3.	./bessel_zeros
4.	python bessel_zeros.py

In a step 3, a possibly hugh temporary file *bessel_zeros.txt* will be generated.



