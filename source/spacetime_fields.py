'''
author: Adam Rupe
email: atrupe@ucdavis.edu
liscense: BSD
'''

from itertools import product
from .CAs import *
import numpy as np


def binary_field_ensemble(time, space, number_of):
    '''

    '''
    fields = []
    for i in range(number_of):
        array = np.random.choice([0,1], (time,space))
        fields.append(array)

    return fields

###################################################################################################

def wildcard_tiling(t,x):
    '''
    Returns spacetime field of dimension (x,t) sampled from 0-wildcard tiling language.
    '''
    field = np.zeros((t,x), dtype=np.uint8)
    for i in range(t):
        for j in range(x):
            if i % 2 == 0 and j % 2 == 0:
                field[i,j] = np.random.choice([0,1])
            elif i % 2 == 1 and j % 2 == 1:
                field[i,j] = np.random.choice([0,1])

    return field

###################################################################################################

def wildcard_tiling_generator(time, space, number_of):
    '''

    '''
    for i in range(number_of):
        array = wildcard_tiling(time, space)
        yield array

###################################################################################################

def domain_18_ensemble(time, space, number_of, wildcard='even'):
    '''

    '''
    fields = []
    domain = ECA(18, domain_18(space, wildcard=wildcard))
    for i in range(number_of):
        domain.reset(domain_18(space, wildcard=wildcard))
        domain.evolve(time)
        fields.append(domain.spacetime)
    return fields

###################################################################################################

def domain_18_generator(time, space, number_of, wildcard='even'):
    '''
    Generator for instances of the rule 18 domain. Yields given number of domain 18
    spacetime fields with the given shape.
    '''
    domain = ECA(18, domain_18(space, wildcard=wildcard))
    for i in range(number_of):
        domain.reset(domain_18(space, wildcard=wildcard))
        domain.evolve(time - 1)
        yield domain.spacetime

###################################################################################################

def domain_22_ensemble(time, space, number_of):
    '''

    '''
    fields = []
    domain = ECA(22, domain_22(space))
    for i in range(number_of):
        domain.reset(domain_22(space))
        domain.evolve(time - 1)
        fields.append(domain.spacetime)
    return fields

###################################################################################################

def domain_22_generator(time, space, number_of):
    '''

    '''
    domain = ECA(22, domain_22(space))
    for i in range(number_of):
        domain.reset(domain_22(space))
        domain.evolve(time - 1)
        yield domain.spacetime

###################################################################################################

def ECA_ensemble(rule_number, time, space, number_of, transient=0):
    '''
    Produces an ensemble of spacetime fields for the given ECA rule number evolved
    from different random initial conditions each time. Returns list of the
    spacetime fields.

    Parameters
    ----------
    rule_number: int
        The rule number for the ECA for which multilple fields are desired.

    (time, space): tuple of ints
        The desired shape of the spacetime fields to be produced.

    number_of: int
        The number of spacetime fields desired.

    transient: int, optional (default=0)
        An optional number of initial time steps to remove from the spacetime fields.
        Will still produce fields of the given shape.

    Returns
    -------
    fields: list
        List of the spacetime fields in the ensemble.
    '''
    ca = ECA(rule_number, random_state(space))
    fields = []
    for i in range(number_of):
        ca.reset(random_state(space))
        ca.evolve(time + transient - 1)
        fields.append(ca.spacetime[transient:])
    return fields

def ECA_ensemble_generator(rule_number, time,space, number_of, transient=0):
    '''

    '''
    ca = ECA(rule_number, random_state(space))
    for i in range(number_of):
        ca.reset(random_state(space))
        ca.evolve(time + transient - 1)
        yield ca.spacetime[transient:]

###################################################################################################

def CA_ensemble(rule_number, A, R, time, space, number_of, transient=0):
    '''

    '''
    fields = []
    ca = CA(rule_number, A, R, random_state(space, A))
    for i in range(number_of):
        ca.reset(random_state(space, A))
        ca.evolve(time + transient - 1)
        fields.append(ca.spacetime[transient:])
    return fields

###################################################################################################

def CA_ensemble_generator(rule_number, A, R, time,space, number_of, transient=0):
    '''

    '''
    ca = CA(rule_number, A, R, random_state(space, A))
    for i in range(number_of):
        ca.reset(random_state(space, A))
        ca.evolve(time + transient - 1)
        yield ca.spacetime[transient:]

###################################################################################################

def tri_line_field(reps, space):

    field = np.random.choice([0,1], space)
    np.zeros(space, dtype=int)
    field = np.vstack((field,
                    np.zeros(space, dtype=int),
                    np.zeros(space, dtype=int)))
    for i in range(reps - 1):
        field = np.vstack((field,
                          np.random.choice([0,1], space),
                          np.zeros(space, dtype=int),
                          np.zeros(space, dtype=int),
                          ))
    return field

def tri_line_ensemble(reps, space, number_of):

    ensemble = []
    for i in range(number_of):
        ensemble.append(tri_line_field(reps, space))
    return ensemble

###################################################################################################

def alternating_tile_field(t_reps, x_reps):
    tempA = np.zeros((3,3), dtype=int)
    tempA[1,1] = 1
    tempB = np.ones((3,3), dtype=int)
    tempB[1,1] = 0
    upper = np.hstack((tempA,tempB))
    lower = np.hstack((tempB, tempA))
    full_template = np.vstack((upper,lower))

    return np.tile(full_template, (t_reps, x_reps))

###################################################################################################
