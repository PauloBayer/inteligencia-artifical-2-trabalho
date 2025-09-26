"""
Genetic Algorithm implemetantion
================================

This Python module contains an implementation of a genetic algorithm based on
binary representation.

The code is based on the MATLAB implementation developed by Roberto T. Raittz.
It has been adapted to Python by Diogo de J. S. Machado.

Functions
---------
The module includes the following functions:

- `gago`: Genetic algorithm for optimization, based on binary representation.
- `randipow`: Generate a matrix of random integers raised to a fixed value.

Authorship
----------
Original implementention author: Roberto T. Raittz

Python version author: Diogo de J. S. Machado

Date
----
April 6th, 2024

"""

import numpy as np

def gago(ffit, nbits, gaoptions={}):
    """
    Genetic algorithm for optimization.

    Parameters
    ----------
    - ffit (function): Fitness function to be optimized.
    - nbits (int): Number of bits in the binary representation of each
      individual.
    - gaoptions (dict, optional): Dictionary containing the following
      parameters:
        - "PopulationSize" (int, optional): Number of individuals in the
          population. Default is 100.
        - "Generations" (int, optional): Number of generations for which the
          algorithm will run. Default is 300.
        - "MutationFcn" (float, optional): Mutation rate (0 to 1). Default is
          0.15.
        - "EliteCount" (int, optional): Number of elite individuals to be
          preserved in each generation. Default is 2.
        - "InitialPopulation" (numpy.ndarray, optional): Initial population.
          If None, a random population will be generated.

    Returns
    -------
    - tuple: A tuple containing:
        - x (numpy.ndarray): The best individual found.
        - popx (numpy.ndarray): The final population.
        - fitvals (numpy.ndarray): Fitness values of the final population.
    """
    m = nbits
    n = gaoptions.get("PopulationSize", 100)
    nger = gaoptions.get("Generations", 300)
    mutrate = gaoptions.get("MutationFcn", 0.15)
    nelite = gaoptions.get("EliteCount", 2)
    pop0 = gaoptions.get("InitialPopulation", None)

    # Generate initial population if not provided
    if (pop0 is None or len(pop0) == 0):
        pop0 = np.random.randint(2, size=(n, m)).astype(np.uint8)

    nmut = np.int_(mutrate * n)
    popx = pop0
    fits = np.zeros(n)

    # Main loop for genetic algorithm
    for j in range(0, nger):
        # Evaluate fitness of each individual in the population
        for i in range(0, n):
            fits[i] = ffit(popx[i])

        # Sort fits and get sorted indices
        iord = np.argsort(fits)

        # Reorder popx according to sorted indices
        popx = popx[iord, :]

        # Create new population as a copy of popx
        newpop = np.copy(popx)

        # Crossover
        i1 = randipow(n, 1.3, n, 1).flatten()
        i2 = randipow(n, 1.3, n, 1).flatten()

        pop1 = popx[i1, :]
        pop2 = popx[i2, :]

        icros = np.random.randint(m, size=(n, 1))
        ids = np.tile(np.arange(1, m + 1), (n, 1))
        crt = ids > icros

        newpop = np.copy(popx)
        newpop[crt] = pop1[crt]
        newpop[~crt] = pop2[~crt]

        # Mutation
        if nmut > 0:
            popmut = newpop[:nmut]

            shape_original = np.shape(popmut)
            popmut = popmut.flatten('F')

            ids = np.where(popmut | ~popmut)[0]
            xmut = np.random.randint(len(ids), size=nmut)
            popmut[xmut] = 1 - popmut[xmut]

            popmut = popmut.reshape(shape_original, order='F')

            # Elitism
            if nelite > 0:
                newpop[:nelite, :] = popx[:nelite, :]

            # Make new population
            newpop[nelite:nelite + nmut, :] = popmut
            popx = newpop

    # Return results
    iord = np.argsort(fits)
    fitvals = fits[iord]
    popx = popx[iord, :]
    x = popx[0, :]
    
    return x, popx, fitvals


def randipow(xmax, xpow, n, m):
    """
    Generate a matrix of random integers raised to a fixed value.

    Parameters
    ----------
    - xmax (float): The maximum value to scale the random numbers by.
    - xpow (float): The fixed value to which each random number will be raised.
    - n (int): The number of rows in the resulting matrix.
    - m (int): The number of columns in the resulting matrix.

    Returns
    -------
    - numpy.ndarray: A matrix of integers with dimensions (n, m).

    Reference
    ---------
    - http://www.sbgames.org/papers/sbgames08/posters/papers/p19.pdf
    """
    # Generate a matrix of uniformly distributed random numbers between 0 and 1
    rand_nums = np.random.rand(n, m)
    
    # Apply the power xpow to each element of the random number matrix
    powered_rand_nums = rand_nums ** xpow
    
    # Multiply by xmax and convert to integers truncated towards zero
    mret = np.int_(xmax * powered_rand_nums)
    
    return mret