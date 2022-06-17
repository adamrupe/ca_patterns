'''
General 1+1 D CA Simulator Code

author: Adam Rupe
email: atrupe@ucdavis.edu
liscense: BSD
'''

import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
from itertools import product
from numba import njit


@njit
def numba_update(spacetime, lookup_table, time):
    '''
    Test with numba. Only for radius 1 at the moment.

    '''
    lattice_size = spacetime.shape[1]
    current_state = spacetime[0]
    table = lookup_table

    #new_spacetime = np.zeros((time, lattice_size), dtype=np.uint8)
    new_spacetime = np.zeros((time, lattice_size), dtype=spacetime.dtype)
    for t in range(time):
        new_state = new_spacetime[t]
        for i in range(lattice_size):
            # this b.c. check is more efficient than modulo operator
            if i == lattice_size - 1:
                j = 0
            else:
                j = i+1
            neighborhood = (current_state[i-1],
                            current_state[i],
                            #current_state[(i+1)%lattice_size])
                            current_state[j])
            new_state[i] = table[neighborhood]
            #new_spacetime[t,i] = table[neighborhood]
        current_state = new_state

    spacetime = np.concatenate((spacetime, new_spacetime))

    return spacetime

class CA(object):
    '''
    Class for 1+1 D deterministic synchronus cellular automaton simulator.

    '''
    def __init__(self, rule, alphabet_size, radius, initial_condition):

        '''
        Initializes the CA.

        --To add--
        use max_rule() function to raise exception if rule number is not
        consistent with given radius and alphabet size

        come up with a way to give a lookup table as input for initializing
        a CA instance

        try-except for initial condition dtype; should be as small of memory
        footprint as possible, unsigned int.

        Parameters
        ----------

        rule: int
            Defines mapping from current neighborhood to next state. Generalization
             of Wolfram ECA rule numbering used.
            Use max_rule(alphabet_size, radius) function to get maximum rule number
            allowed for give alphabet size and radius.
        alphabet_size: int
            Number of local states for each site on CA.
        radius: int
            Size of neighborhood used for update rule. Neighborhood size is 2*radius +1.
        initial_condition: array
            Initializes initial condition for CA.
            *** Could generalize to take any array-like input and convert to np array***

        Instance Variables
        ------------------
        self.initial:
            Stored initial conditions.
        self.spacetime:
            Array of spacetime values. Initialized to initial_condition. Space is
            in horizontal direction, time is vertical.
        self.rule:
            CA lookup table rule. Initialized to rule.
        self.alphabet:
            Local state alphabet size. Initialized to alphabet_size
        self.radius:
            Neighborhood radius. Initialized to radius.
        self.table:
            Hypercube array for CA lookup table using CA rule number. Neighborhoods
            represented as coordinates on hypercube, and value at the coordinate
            is the corresponding mapping of that neighborhood.

        '''
        self.rule = rule
        self.alphabet = alphabet_size
        self.radius = radius
        self.table = lookup_table(rule, alphabet_size, radius)
        if initial_condition is None:
            self.initial = None
            self.lattice_size = 0
            self.current_state = None
            self.spacetime = None
        else:
            self.initial = np.copy(initial_condition).astype(np.uint8)
            self.lattice_size = np.size(initial_condition)
            self.current_state = np.copy(initial_condition).astype(np.uint8)
            # use np.newaxis to make spacetime two dimensional
            self.spacetime = np.copy(self.initial)[np.newaxis, ...]


    def evolve(self, time):
        '''
         Evolves the CA for specified number of steps.

        Parameters
        ---------
        time: int
            Number of time steps to evolve the CA.

        Updates
        -------
        self._spacetime
            Adds new spacetime data to array of spacetime data.

        '''
        new_spacetime = np.empty((time, self.lattice_size), dtype=np.uint8)
        for t in range(time):
            # Get array of current neighborhood values
            neighbors = neighborhood(self.current_state, self.radius)
            # Run the current neighborhood values through the lookup table to get next configuration
            self.current_state = self.table[neighbors]
            # Update current state of the CA
            new_spacetime[t] = self.current_state

        # Add the new spacetime field to the original field
        self.spacetime = np.concatenate((self.spacetime, new_spacetime))

    def numba_evolve(self, time):
        '''
        Test with numba. Only for radius 1 at the moment.
        '''
        self.spacetime = numba_update(self.spacetime, self.table, time)
        self.current_state = self.spacetime[-1]


    def fast_evolve(self, time):
        '''
        Evolves the CA state only and does not store the spacetime field
        '''
        for t in range(time):
            #get array of current neighborhood values
            neighbors = neighborhood(self.current_state, self.radius)
            #initialize array for new configuration string
            next_state = np.zeros(self.lattice_size, dtype=np.uint8)
            #run the current neighborhood values through the lookup table to get next configuration
            next_state = self.table[neighbors]
            #add new configuration values to spacetime array
            self.current_state = next_state

    def reset(self, new_state=None):
        '''
        Resets the spacetime data back to the given state or original initial conditions.

        Parameters
        ----------
        new_state: array_like, optional (default = None)
            Configuration of new initial conditions to reset the CA to.
            If None, CA is returned to original initial conditions.
        '''
        if new_state is None:
            #reset spacetime data to the initial conditions
            self.current_state = np.copy(self.initial)
            self.spacetime = np.copy(self.initial)[np.newaxis, ...]
        else:
            state = np.array(new_state).astype(np.uint8)
            self.lattice_size = np.size(state)
            self.initial = np.copy(state)
            self.current_state = np.copy(state)
            self.spacetime = np.copy(state)[np.newaxis, ...]

    def rewind(self, steps):
        '''
        Rewind spacetime data the given number of steps from current configuration.

        Parameters
        ----------
        steps: int
            Number of time steps to rewind.
        '''
        rows, columns = np.shape(self.spacetime)
        self.spacetime = self.spacetime[:rows - steps]
        self.current_state = self.spacetime[rows-steps-1]

    def lookup_table(self, full=False):
        '''
        Provides the lookup table for the CA.

        Parameters
        ----------
        full: bool, optional (default = False)
            If True, will print full lookup table in lexicographic order. Best
            if used for ECAs and other small neighborhood size rules.

            If False, returns the lookup table as a dict.

        '''
        #simplify variable names. R = neighborhood size
        A = self.alphabet
        R = 2*self.radius + 1
        scan = tuple(np.arange(0,A)[::-1])
        if full:
            for a in product(scan, repeat = R):
                print(a , self.table[a])

            return None
        else:
            lookup = {a : self.table[a] for a in product(scan, repeat = R)}
            return lookup

    def perturb(self, N_flips):
        '''
        Randomly flips specified number of bits on current spatial configuration of CA.

        FIX:
        Add condition for when ECA is instantiated and perturbed before evolving
        -- rows,columns = np.shape(...) fails in this case

        Parameters
        ----------

        N_flips: int
            Number of sites on spatial configuration to flip

        Updates
        -------
        self._spacetime
            Last row of spacetime array (current configuration) has 'N_flips'
            bits randomly selected and flipped.

        '''
        rows, columns  = np.shape(self.spacetime)
        #indices for current configuration array
        indices = np.arange(0, columns, dtype=np.uint8)
        #array to keep track of location of bit flips so a single site is not flipped multiple times
        flip_check = np.zeros(columns)
        count = 0
        #loop to flip bits while making sure to not flip bits more than once
        while count < N_flips:
            flip = np.random.choice(indices)
            if flip_check[flip] == 0:
                flip_check[flip] += 1
                count += 1
                self.spacetime[rows -1][flip] = -self.spacetime[rows -1][flip] + 1

class ECA(CA):
    '''
    Elementary cellular automata subclass. Same as CA subclass but with alphabet size of 2 and
    neighborhood radius 1.
    '''
    alphabet_size = 2
    radius = 1

    def __init__(self, rule, initial_condition):
        '''
        Initializes the ECA.

        Parameters
        ----------
        rule: int between 0 and 255
            Defines the lookup table using Wolfram lexicographic numbering scheme.

        initial_condition: array_like
            Initial configuration for the ECA.
        '''
        super(ECA, self).__init__(rule, ECA.alphabet_size, ECA.radius, initial_condition)


''' Support Code Below'''

def max_rule(alphabet, radius):
    '''
    Returns the maximum rule number for a given alphabet size and neighborhood radius.

    Parameters
    ----------
    alphabet: int
        Alphabet size for the CA.

    radius: int
        Neighborhood radius length for the CA.

    Returns
    -------
    rule number: int
        The maximum allowed rule number for the given alphabet size and radius.
    '''
    rule = 0
    #R = neighborhood size
    R = 2*radius + 1
    #add up max possible output for every neighborhood
    for i in range(alphabet ** R):
        rule += (alphabet - 1) * (alphabet)**i

    return rule


def random_rule(alphabet_size, radius):
    '''
    Randomly generates a rule number for a CA with the given alphabet size and radius.

    Parameters
    ----------
    alphabet_size: int
        Alphabet size for the CA.

    radius: int
        Neighborhood radius length for the CA.

    Returns
    -------
    rule_number: int
        A rule number, following the Wolfram numbering scheme, for the CA lookup table
        which is consistent with the given alphabet size and radius.
    '''
    A = alphabet_size
    R = radius
    max_ = max_rule(A, R)
    max_rule_digits = int(math.log10(max_)) + 1
    good_number = False
    while not good_number:
        number_of_digits = np.random.randint(1, max_rule_digits)
        rule_number_array = np.random.randint(0, 9, size=number_of_digits)
        rule_number = int(''.join(str(a) for a in rule_number_array))
        if rule_number <= max_:
            good_number = True

    return rule_number


def rule_number_conversion(rule_number, alphabet, radius):
    '''
    Returns integer in numerical base of the alphabet, padded with zeros to
    properly fill out neighborhood size.

    Parameters
    ----------
    rule_number: int
        Integer for the rule number.

    alphabet: int
        Alphabet size for the CA.

    radius: int
        Size of the radius for the CA look up table neighborhood.

    Returns
    -------
    out: str
        String for rule number expanded in base of the alphabet size as to properly
        match neighborhoods in lexicographical order for produce the CA lookup table.
    '''
    length = alphabet ** (2*radius + 1)
    unformatted_conversion = np.base_repr(rule_number, alphabet)
    return '{:{fill}{align}{width}}'.format(unformatted_conversion,
                                            fill='0',
                                            align='>',
                                            width=length)


def lookup_table(rule, alphabet_size, radius):
    '''
    Produces the lookup table for the specified CA rule of the given alphabet
    size and radius.

    Parameters
    ----------
    rule: int
        CA rule number that defines the lookup table. For neighborhood in lexicographic
        order, the corresponding CA update rule mapping is added up in the base
        of the alphabet size. This is the rule number.

    alphabet_size: int
        Number of local site states available to the CA.

    radius: int
        Radius of the neighorhood used for the update rule. For radius R, neighborhood
        size is 2R+1.


    Returns
    -------
    table: ndarray
        Multidimensional array representation of lookup table as a hypercube. The
        neighborhood is represented as coordinates on the hypercube and the associated
        update mapping of that neighborhood is the value of the hypercube at those
        coordinates.
    '''
    R = 2*radius + 1
    A = alphabet_size
    #convert rule number to appropriate base using rule_number_conversion function
    rule_number = rule_number_conversion(rule, alphabet_size, radius)
    #build shape tuple to get corret shape for lookuptable ndarray from the given radius and alphabet size
    shape = tuple(A * np.ones(R, dtype=int))
    #initialize empty ndarray for table with the correct shape
    table = np.zeros(shape, dtype=np.uint8)
    #scan is tuple (A, A-1, A-2, ..., 0) used to build neighborhoods in lexicographic order
    scan = tuple(np.arange(0,A)[::-1])
    #run loop through neighborhoods in lexicographic order and fill in the lookup table outputs
    #on the ndarray at coordinates corresponding to the neighborhood value
    for a, rule_val in zip(product(scan, repeat = R), rule_number):
        table[a] = rule_val

    return table


def neighborhood(data, radius):
    '''
    Returns list of arrays of neighborhood values for input data. Formatted to
    index the multidimensional array lookup table.

    Parameters
    ----------
    data: array_like
        Input string of data.

    radius: int
        Radius of the neighborood.

    Returns
    -------
    neighborhood: list of arrays
        List of neighbor arrays formatted to index multidimensional array lookup
        table. Of the form [array(leftmost neighbors), ..., array(left nearest neighbors),
        array(data), array(right nearest neighbors), ..., array(rightmost neighbors)]

    '''
    #initialize lists for left enighbor values and right neighbor values
    left = []
    right = []
    #run loop through radial values up to the radius. left values are run in reverse so that both sides build
    #from inside out. r = 0 is skipped and added on in the last step
    for r in range(radius):
        left += [np.roll(data, radius-r)]
        right += [np.roll(data, -(r+1))]
    neighbors = left + [np.array(data)] + right
    return tuple(neighbors)

@njit
def numba_spacetime(initial, lookup_table, time):
    '''
    Test with numba. Only for radius 1 at the moment.

    Parameters
    ----------
    initial: array-like
        String of data used for initial condition.

    lookup_table: ndarray
        Lookup table in ndarray form. Created using the lookup_table function.

    time: int
        Number of time steps to evolve

    Returns
    -------
    spacetime: ndarray
        2D numpy array of spacetime field. Time is the zeroth (vertical) axis,
        space the first (horizontal).

    '''
    current_state = np.copy(initial).astype(np.uint8)
    lattice_size = current_state.size
    table = lookup_table

    # initialze spacetime field and set first configuration to the initial config
    spacetime = np.zeros((time+1, lattice_size), dtype=np.uint8)
    spacetime[0] = current_state
    # update the spacetime field for the given time steps
    for t in range(time):
        new_state = spacetime[t+1]
        for i in range(lattice_size):
            # this b.c. check is more efficient than modulo operator
            if i == lattice_size - 1:
                j = 0
            else:
                j = i+1
            neighborhood = (current_state[i-1],
                            current_state[i],
                            current_state[j])
            new_state[i] = table[neighborhood]
        # update the current state as the new state
        current_state = new_state

    return spacetime


'''Initial Condition Creation Code Below'''


def random_state(length, alphabet_size=2):
    '''
    Returns array of randomly selected states of given length with given alphabet size.

    Parameters
    ----------
    length: int
        Lenght of the array

    alphabet_size: int, optional (default=2)
        Size of the alphabet to be randomly selected from in creating array.

    Returns
    -------
    state: array
        Array of given length of states randomly selected from {0,...,A -1} for
        alphabet size A.
    '''
    #state = np.random.randint(alphabet_size, size=length, dtype=np.uint8)
    state = np.random.choice(np.arange(alphabet_size), length).astype(np.uint8)
    return state


def one_one(length):
    '''
    Returns an array of the given length with all zeros, except for a single 1 in
    the center.
    '''
    array = np.zeros(length, dtype=np.uint8)
    array[int(length/2)] = 1
    return array


def uniform_state(length, symbol):
    '''
    Returns homogeneous array of given symbol at given length

    Parameters
    ----------
    length: int
        Length of the array to be created.

    symbol: int
        Symbol which will uniformly populate the array.

    Returns
    -------
    out: array
        Homogeneous array of given length uniformly populated by given symbol.
    '''
    if symbol == 1:
        state = np.ones(length, dtype=np.uint8)
    elif symbol == 0:
        state = np.zeros(length, dtype=np.uint8)
    else:
        state = symbol * np.ones(length, dtype=np.uint8)

    return state


def variable_density_state(length, p):
    '''
    Returns array of 1s and 0s with the 1s distributed as p and 0s 1-p.

    Parameters
    ----------
    length: int
        Total length of the array.

    p: float
        Distribution of 1s in the array

    Returns
    -------
    out: array
        Np array of 1s and 0s.
    '''
    state = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        if np.random.rand() < p:
            state[i] = 1
    return state


def n_ones(length, n ):
    '''
    Array of given length with the given number of 1s randomly distributed among 0s.

    Parameters
    ----------
    legnth: int
        Total length of the array.

    n: int
        Number of 1s to be randomly distributed among length - n 0s.

    Returns
    -------
    out: array
        Array of n 1s randomly distributed among length - n 0s.
    '''
    state = np.zeros(length, dtype=np.uint8)
    ind = np.arange(0,length,1)
    count = 0
    while count < n:
        test = np.random.choice(ind)
        if state[test] == 0:
            state[test] =1
            count += 1

    return state


def wildcard(bias):
    return np.random.choice([0,1], p=[1-bias, bias]).astype(np.uint8)


def domain_18(length, bias = 0.5, wildcard = 'even'):
    '''
    Returns array sampled from the rule 18 zero-wildcard domain language.

    Parameters
    ----------
    length: int
        Total length of the array

    bias: float, optional (default = 0.5)
        Probability bias for the wildcard state producing a 1.

    wildcard: str, optional (default = 'even')
        Determines the phase of the domain language. 'even' sets the wildcard
        states on even lattice sites, 'odd' sets the wildcard states on odd lattice sites.
    '''
    state = np.zeros(length, dtype=np.uint8)
    for s in range(length):
        if wildcard == 'even':
            if s % 2 == 0 and np.random.rand() < bias:
                state[s] = 1
        elif wildcard == 'odd':
            if (s+1) % 2 == 0 and np.random.rand() < bias:
                state[s] = 1

    return state


def one_one_zero_wildcard(length, start_state=0, wildcard_bias=0.5):
    '''
    Code for generating strings of the one-one-zero-wildcard language; one of two
    chaotic domains for A=2, R=2, CA rule 2614700074. Crutchfild Hanson "Turbulent
    Pattern Bases for Cellular Automata".
    '''
    state = start_state
    output=np.zeros(length,dtype=np.uint8)
    for i in range(length):
        state, output[i] = {0:[1,1],
                            1:[2,1],
                            2:[3,0],
                            3:[0,wildcard(bias=wildcard_bias)]}[state]
    return output


def domain_54(length, phase='a'):
    '''
    Returns array sampled from the rule 54 domain language.

    Parameters
    ----------
    length: int
        Total length of the array. Should be a multiple of 4 to produce a pure-domain
        field using periodic boundary conditions.

    phase:
        Starting phase of the domain machine that will generate the array.
        Follows machine convention from
        Hanson and Crutchfield Physica D 1997.

    Returns
    -------
    out: array
        Array sampled from the rule 54 domain language, sampled from the domain
        machine starting in the given phase.
    '''
    generator = {'a':('b', 0) , 'b':('c', 0) , 'c':('d', 0) , 'd':('a', 1) ,
                 'e':('f', 1) , 'f':('g', 1) , 'g':('h', 1) , 'h':('e', 0)}

    machine_state = phase
    output = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        machine_state , output[i] = generator[machine_state]

    return output


def domain_22(length):
    '''
    Returns array sampled from the 0-0-0-wildcard phase of the rule 22 domain
    language.

    Parameters
    ----------
    length: int
        Desired length of the output array. Should be a multiple of 4 to produce
        a pure-domain field with periodic boundary conditions.

    Returns
    -------
    out: array
        Array sampled from one phase of the rule 22 domain.
    '''
    state = np.zeros(length, dtype=np.uint8)
    for s in range(length):
        if s % 4 == 0 and np.random.rand() < 0.5:
            state[s] = 1

    return state
