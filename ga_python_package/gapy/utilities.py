"""
Utility functions
=================

This Python module contains some utility functions for data manipulation and
transformation.

The code is based on the MATLAB implementation developed by Roberto T. Raittz.
It has been adapted to Python by Diogo de J. S. Machado.

Functions
---------
The module includes the following functions:

- `mat2vec`: Convert a matrix to a 1D array by flattening it row-wise.
- `vec2mat`: Create a matrix from a 1D array, arranging its elements row-wise.
- `bytes2bits`: Convert a list of numbers or a single number to a bit array.
- `bits2bytes`: Convert a bit array to bytes.

Authorship
----------
Original implementention author: Roberto T. Raittz

Python version author: Diogo de J. S. Machado

Date
----
April 6th, 2024

"""

import numpy as np

def mat2vec(R):
    """
    Convert a matrix to a 1D array by flattening it row-wise.

    Parameters
    ----------
    - R (numpy.ndarray): Input matrix.

    Returns
    -------
    - numpy.ndarray: Flattened 1D array containing elements of the input
      matrix.
    """
    # Flatten matrix by rows
    mret = np.array(R).flatten('C')
    return mret

def vec2mat(vect, larg):
    """
    Creates a matrix with values of 'vect' row-wise with 'larg' columns 
    (completes the last row with zeros if necessary).

    Parameters
    ----------
    - vect (list): Input vector.
    - larg (int): Number of columns in the resulting matrix.

    Returns
    -------
    - numpy.ndarray: Matrix created from the input vector.
    """
    nv = len(vect)
    n = np.ceil(nv / larg).astype(int)
    mat = np.zeros((n, larg))
    mat_flat = mat.flatten()
    mat_flat[:nv] = vect
    mat = mat_flat.reshape((n, larg))
    return mat

def bytes2bits(s, dtype='uint8'):
    """
    Convert a list of numbers or a single number to a bit array.

    Parameters
    ----------
    - s (list or int or float): Input list of numbers or a single number.
    - dtype (str, optional): Data type of the input numbers. Default is
      'uint8'.

    Returns
    -------
    - numpy.ndarray: Bit array containing the binary representation of the
      input numbers.
    """
    # Convert single number to a list with one element
    if isinstance(s, int) or isinstance(s, float):
        s = [s]

    # Convert the input list to a NumPy array of dtype
    s = np.array(s)
    s_bytes = np.array(s, dtype=dtype).view(np.uint8)

    # Convert bytes to bits and reshape
    s_bits = np.unpackbits(s_bytes, bitorder='little')

    # Convert array elements to int
    s_bits = s_bits.astype(int)

    return s_bits

def bits2bytes(xbits, dtype='uint8'):
    """
    Convert a bit array to bytes.

    Parameters
    ----------
    - xbits (numpy.ndarray): Input bit array.
    - dtype (str, optional): Data type of the output bytes. Default is 'uint8'.

    Returns
    -------
    - numpy.ndarray: Byte array containing the byte representation of the
      input bit array.
    """
    # Pack the bits into bytes
    bytes_array = np.packbits(xbits, bitorder='little')

    # Convert the bytes array to the specified data type
    bytes_array = bytes_array.astype(np.uint8).view(dtype=dtype)

    return bytes_array