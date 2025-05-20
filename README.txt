For the implementation of the Friedman kinetic analysis code, please consider the following:

1.- Edit the sections marked as 'Edit---' to update the file names and the corresponding heating rates for each dataset.

2.- When running the program, you will be prompted in the console to enter the desired smoothing level. 
For smoothing the first derivative, a value between 21 and 51 is recommended. For the second smoothing step, applied
to the natural logarithm, values of 301 or higher are advisable.

3.- Upon completion, the code will generate five output files:

		data_friedman.txt: Contains the data points used for the Friedman linear fitting.

		resultsfriedman.txt: Includes the calculated activation energies for each degree of conversion.

		weightvstemp.txt: Provides the mass loss curves as a function of temperature.

		dtgvstemp.txt: Contains the first derivative curves (DTG) of mass versus temperature.

		kinetics.jpg: A summary image of the overall kinetic analysis.