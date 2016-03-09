import numpy as np

def smoothify(thisarray):
    '''
    returns moving average of input using:
    out(n) = .7*in(n) + 0.15*( in(n-1) + in(n+1) )
    '''

    # make sure we got a numpy array, else make it one
    if type(thisarray) == type([]): thisarray = np.array(thisarray)

    # do the moving average by adding three slices of the original array
    # returns a numpy array,
    # could be modified to return whatever type we put in...
    return 0.7 * thisarray[1:-1] + 0.15 * ( thisarray[2:] + thisarray[:-2] )

