'''
Support code to be used by topological and probabilistic modules. Most of this
code is used (or can be used) by both modules, but not all.

author: Adam Rupe
email: atrupe@ucdavis.edu
liscense: BSD
'''
from itertools import product
from numba import jit, njit
from collections import defaultdict

import numpy as np
# import cPickle as pickle

###################################################################################################

def lightcone_size(depth, c):
    '''
    Computes lenght of random vector representation of light cone configuration.
    Technically this is the past lightcone depth, as it includes the present site.
    Subtract 1 from this for future lightcone sides.

    Parameters
    ----------
    depth: int
        Depth of the light cone under consideration. Depth = 0 is just the current
        site. Depth = 1 includes one time step into the past, etc.

    c: int
        Speed of information propagation in the spacetime system.

    Returns
    -------
    size: int
        Length of light cone configuration vector.
    '''
    size = 0
    for d in range(depth+1):
        size += 2*c*d + 1
    return size

###################################################################################################

def plc_shape(plc_array, depth, c):
    '''
    Converts past light cone configuration random vector into string representation of actual
    light cone shape. For 1+1 D lightcones.

    Parameters
    ----------
    plc_array: array_like
        Array of past light cone configuration random vector.

    depth: int
        Depth of light cone that is being converted.

    c: int
        Speed of information propogation in spacetime system.

    Returns
    -------
    shape: str
        String that when printed gives actual past light cone shape.
    '''
    shape = ''
    tracer = len(plc_array)
    for d in range(depth + 1):
        D = depth - d
        window = 2*D*c + 1
        shape += c*d*' '
        tracer -= window
        for w in range(window):
            shape += str(plc_array[tracer + w])
        shape += c*d*' ' + '\n'

    return shape

###################################################################################################

def flc_shape(flc_array, depth, c):
    '''
    Converts future light cone configuration random vector into string representation of actual
    light cone shape. For 1+1 D lightcones.

    Parameters
    ----------
    plc_array: array_like
        Array of future light cone configuration random vector.

    depth: int
        Depth of light cone that is being converted.

    c: int
        Speed of information propogation in spacetime system.

    Returns
    -------
    shape: str
        String that when printed gives actual future light cone shape.
    '''

    shape = ''
    tracer = 0
    for step in range(depth):
        d = step + 1
        D = depth - d
        window = 2*d*c + 1
        shape += c*D*' '
        for w in range(window):
            shape += str(flc_array[tracer ])
            tracer += 1
        shape += c*D*' ' + '\n'

    return shape

###################################################################################################

def V_f_size(past_depth, V_depth, c):
    '''
    Returns the size of the configuration array for V of the given depth and
    propagation speed.
    '''
    upper = 2*V_depth*c*(past_depth + 1)
    lower = V_depth
    for d in range(V_depth):
        lower += 2*c*d
    return upper + lower

###################################################################################################

def plc_gen(position, past_depth, propagation_speed, shape='full'):
    '''
    position is tuple (t,x)
    '''
    if shape not in ['full', 'left', 'right']:
        raise ValueError("shape must be either 'full', 'right', or 'left' ")

    pd = past_depth
    c = propagation_speed

    t, x = position
    time = []
    space = []

    for d in range(pd + 1):
        if shape == 'full':
            window_size = 2*d*c + 1
        else:
            window_size = d*c + 1
        current_time = []
        current_space = []
        for w in range(window_size):
            current_time += [t - d]
            if shape == 'right':
                a = w
            else:
                a = - d*c + w
            current_space += [x + a]
        time += current_time
        space += current_space
    return [np.array(time), np.array(space)]

###################################################################################################

def flc_gen(position, future_depth, propagation_speed, shape='full'):
    '''
    position is tuple (t,x)
    '''
    if shape not in ['full', 'left', 'right']:
        raise ValueError("shape must be either 'full', 'right', or 'left' ")

    fd = future_depth
    c = propagation_speed

    t, x = position
    time = []
    space = []

    for depth in range(fd):
        d = depth + 1
        if shape == 'full':
            window_size = 2*d*c + 1
        else:
            window_size = d*c + 1
        current_time = []
        current_space = []
        for w in range(window_size):
            current_time += [t + d]
            if shape == 'right':
                a = w
            else:
                a = - d*c + w
            current_space += [x + a]
        time += current_time
        space += current_space
    return [np.array(time), np.array(space)]

###################################################################################################

def V_f_gen(position, past_depth, future_depth, propagation_speed):
    '''
    Position is tuple (t,x)

    Random vector ordering done in same fashion as lightcones. Order goes from
    bottom to top, and in each row left to right. This is done for each "layer",
    layers are ordered from inside out maintaining concatenation property.
    '''
    pd = past_depth
    fd = future_depth
    c = propagation_speed

    t, x = position

    time = []
    space = []

    for i in range(fd):
        #initialize "tip" of V
        current_time = [t + (i+1)]
        current_space = [x]
        for d in range(pd + (i+1)):
            current_time += [t - d + i for j in range(c)]
            current_space += [x - c*d - (j+1) for j in range(c)]
            current_time += [t - d + i for j in range(c)]
            current_space += [x + c*d + (j+1) for j in range(c)]

        time += current_time
        space += current_space
    return [np.array(time), np.array(space)]

def V_t_gen(position, past_depth, future_depth, propagation_speed):
    '''
    Position is tuple (t,x)

    Random vector ordering done in same fashion as lightcones. Order goes from
    bottom to top, and in each row left to right. This is done for each "layer",
    layers are ordered from inside out maintaining concatenation property.
    '''
    pd = past_depth
    fd = future_depth
    c = propagation_speed

    t, x = position

    time = []
    space = []

    for i in range(fd):
        #initialize "tip" of V
        current_time = [t + (i+1)]
        current_space = [x]
        for d in range(pd):
            current_time += [t - d + i for j in range(c)]
            current_space += [x - c*d - (j+1) for j in range(c)]
            current_time += [t - d + i for j in range(c)]
            current_space += [x + c*d + (j+1) for j in range(c)]

        time += current_time
        space += current_space
    return [np.array(time), np.array(space)]

def V_r_gen(position, past_depth, future_depth, propagation_speed):
    '''
    Position is tuple (t,x)

    Random vector ordering done in same fashion as lightcones. Order goes from
    bottom to top, and in each row left to right. This is done for each "layer",
    layers are ordered from inside out maintaining concatenation property.
    '''
    pd = past_depth
    fd = future_depth
    c = propagation_speed

    t, x = position

    time = []
    space = []

    for i in range(fd):
        current_time = []
        current_space = []
        for d in range(pd + 1):
            current_time += [t - d for j in range(c)]
            current_space += [x + c*d + i + j + 1 for j in range(c)]

        time += current_time
        space += current_space
    return [np.array(time), np.array(space)]

def V_l_gen(position, past_depth, future_depth, propagation_speed):
    '''
    Position is tuple (t,x)

    Random vector ordering done in same fashion as lightcones. Order goes from
    bottom to top, and in each row left to right. This is done for each "layer",
    layers are ordered from inside out maintaining concatenation property.
    '''
    pd = past_depth
    fd = future_depth
    c = propagation_speed

    t, x = position

    time = []
    space = []

    for i in range(fd):
        current_time = []
        current_space = []
        for d in range(pd + 1):
            current_time += [t - d for j in range(c)]
            current_space += [x - c*d - i - j - 1 for j in range(c)]

        time += current_time
        space += current_space
    return [np.array(time), np.array(space)]

###################################################################################################

def right_fringe_gen(position, past_depth, propagation_speed):
    '''
    Deprecated, V_r_gen of future depth 1 used instead
    '''
    pd = past_depth
    c = propagation_speed

    t, x = position
    time = []
    space = []

    for d in range(pd + 1):
        window_arm = c*d
        time += [t - d]
        space += [x + window_arm + 1]
    return [np.array(time), np.array(space)]

###################################################################################################

def left_fringe_gen(position, past_depth, propagation_speed):
    '''
    Deprecated, V_l_gen of future depth 1 used instead
    '''
    pd = past_depth
    c = propagation_speed

    t, x = position

    time = []
    space = []

    for d in range(pd + 1):
        window_arm = c*d
        time += [t - d]
        space += [x - window_arm - 1]
    return [np.array(time), np.array(space)]

###################################################################################################

def contract_plc(plc_array, original_depth, desired_depth, c):
    '''
    Takes a given past lightcone array of given depth and propagation speed and returns
    array of past lightcone at the same base position with same propagation speed, but
    depth reduced by 1. For 1+1 D plcs.
    '''
    if not desired_depth < original_depth:
        raise ValueError("Desired depth must be less than the original depth.")
    iterations = original_depth - desired_depth
    contracted = np.copy(plc_array)
    for i in range(iterations):
        base_size = 2*(original_depth-i)*c + 1
        contracted = contracted[:len(contracted) - base_size]
    return contracted

def contract_flc(flc_array, original_depth, desired_depth, c):
    '''
    Takes a given future lightcone array of given depth and propagation speed and returns
    array of future lightcone at the same base position with same propagation speed, but
    depth reduced by 1. For 1+1 D flcs.
    '''
    if not desired_depth < original_depth:
        raise ValueError("Desired depth must be less than the original depth.")
    iterations = original_depth - desired_depth
    contracted = np.copy(flc_array)
    for i in range(iterations):
        base_size = 2*(original_depth-i)*c + 1
        contracted = contracted[:len(contracted) - base_size]
    return contracted

###################################################################################################

def update_topological_morph(morph_map, past, future):
    '''
    Updates the supplied morph map with the given past and future. If input past
    is not currently in the morph_map, adds this past as new key and initializes
    set with input future as the associated val. If input past is already in the
    morph_map, updates the associated val set with the input future.

    *** Deprecated, use of defaultdict(set) is implemented instead ***

    Parameters
    ----------
    morph_map: dict
        Morph map stored as dictionary with past lightcones as keys and sets of
        future lightcones as vals.

    past: tuple
        Past lightcone array stored as tuple (needs to be tuple to work with
        morph_map dictionary).

    future: tuple
        Future lightcone array stored as tuple (needs to be tuple to work with
        morph_map dictionary).
    '''
    try:
        morph_map[past].add(future)
    except KeyError:
        morph_map[past] = {future}

def merge_topological_morphs(main_map, new_map):
    '''
    Merges the new_map into the main_map. Goes through all the keys (pasts) of the
    new_map and adds them as new keys with associated vals (sets of futures)
    if they are not already in the main, or updates the associated vals if they
    are in the main.

    Parameters
    ----------
    main_map: dict
        Morph map stored as dictionary with past lightcones as keys and sets
        of future lightcones as vals.

    new_map: dict
        Morph map stored as dictionary with past lightcones as keys and sets
        of future lightcones as vals.
    '''
    for plc, new_morph in new_map.items():
        main_map[plc] |= new_morph

def hierarchical_topological_merge(main_map, local_map, excepted_plcs=set()):
    '''
    Merges the local_map into the main_map. Goes through all the keys (pasts) of the
    local_map and adds them as new keys with associated vals (sets of futures)
    if they are not already in the main, or updates the associated vals if they
    are in the main.

    Parameters
    ----------
    main_map: dict
        Morph map stored as dictionary with past lightcones as keys and sets
        of future lightcones as vals.

    local_map: dict
        Morph map stored as dictionary with past lightcones as keys and sets
        of future lightcones as vals.

    excepted_plcs: list, optional (default=[])
        List of past lightcones to be ignored during merging. Default is empty
        list.
    '''
    for plc, new_morph in local_map.items():
        if plc in excepted_plcs:
            pass
        else:
            main_map[plc] |= new_morph

###################################################################################################

def update_probabilistic_morph(dist_map, past, future):
    '''
    Updates the given morph map. The morph map is a dictionary that maps past lightcone
    configuration tuple keys to dictionary values that represent the morph associated
    with that past lightcone. The morph dictionary here is just a bin count, i.e.
    non-normalized, version of the morph.
    '''
    try:
        dist = dist_map[past]
        dist[future] += 1

    # if plc is not in dist_map, update dist map with plc as key and defaultdict
    # as val for the conditional distribution, where the default count is 0.
    except KeyError:
        new_dist = defaultdict(lambda: 0)
        new_dist[future] += 1
        dist_map.update({past: new_dist})

def merge_probabilistic_morphs(main_dist, local_dist):
    '''
    Merges the local map into the main map.

    To be used in parallel with the above update_probabilistic_morph function,
    where the morphs are given as defaultdicts with default count 0.
    '''
    for past, new_morph in local_dist.items():
        try:
            main_morph = main_dist[past]

            for future, count in new_morph.items():
                # since the morphs are defaultdicts, if future is not in main_morph
                # the defaultdict will add it with val 0, then counts will be added on.
                main_morph[future] += count

        except KeyError:
            main_dist.update({past: new_morph})

def merge_morph_counts(main_morph, local_morph):
    '''
    Merges the non-normalized main_morph with the non-normalized local_morph.
    That is, the counts of futures in the local_morph are added to the counts of
    futures in the main_morph.
    '''
    for future, count in local_morph.items():
        try:
            main_morph[future] += count
        except KeyError:
            main_morph.update({future: count})

###################################################################################################

class Extractor(object):
    '''
    Class for extracting lightcones, Vs, and fringes from spacetime fields.

    Attributes
    ----------

    '''

    def __init__(self, past_depth, future_depth, propagation_speed, future_shape='flc',
                 plc_shape='full', flc_shape='full'):
        '''

        '''
        self.past_depth = past_depth
        self.future_depth = future_depth
        self.c = propagation_speed
        self.future_shape = future_shape
        if future_shape not in ['flc', 'V_f', 'V_r', 'V_l', 'V_t']:
            raise ValueError("future_shape must be either 'flc', 'V_f', 'V_r', 'V_l', or 'V_t'")

        # self._block_height = self.past_depth + self.future_depth + 1
        if future_shape == 'flc':
            max_depth = max(self.past_depth, self.future_depth)
            self._block_width = 2*max_depth*self.c + 1
        elif future_shape in ['V_f', 'V_r', 'V_l']:
            self._block_width = 2*self.past_depth*self.c + 1 + 2*self.c*self.future_depth

        else:
            self._block_width = 2*self.past_depth*self.c + 1

        # Amount of padding needed to handle periodic boundary conditions in space
        # so that lightcones at the edge of the field can be gathered
        self._padding = int((self._block_width - 1) / 2)
        # Starting point in top left corner (excluding margin)
        base_anchor = (self.past_depth, self._padding)
        # Create the initial templates to cut out lightcones etc.
        self._base_plc_t, self._base_plc_x = plc_gen(base_anchor, self.past_depth, self.c, shape=plc_shape)
        if self.future_shape == 'flc':
            self._base_future_t, self._base_future_x = flc_gen(base_anchor, self.future_depth, self.c, shape=flc_shape)
        elif self.future_shape == 'V_f':
            self._base_future_t, self._base_future_x = V_f_gen(base_anchor, self.past_depth, self.future_depth, self.c)
        elif self.future_shape == 'V_l':
            self._base_future_t, self._base_future_x = V_l_gen(base_anchor, self.past_depth, self.future_depth, self.c)
        elif self.future_shape == 'V_r':
            self._base_future_t, self._base_future_x = V_r_gen(base_anchor, self.past_depth, self.future_depth, self.c)
        else:
            self._base_future_t, self._base_future_x = V_t_gen(base_anchor, self.past_depth, self.future_depth, self.c)

        # block_anchor = (self.past_depth, int((self._block_width - 1)/2))
        # self._plc_extractor = plc_gen(block_anchor, self.past_depth, self.c)
        # self._flc_extractor = flc_gen(block_anchor, self.future_depth, self.c)
        # self._V_extractor = V_f_gen(block_anchor, self.past_depth, self.future_depth, self.c)

        self._right_fringe_t, self._right_fringe_x = V_r_gen(base_anchor, self.past_depth, 1, self.c)
        self._left_fringe_t, self._left_fringe_x = V_l_gen(base_anchor, self.past_depth, 1, self.c)
        self._forward_fringe_t, self._forward_fringe_x = V_t_gen(base_anchor, self.past_depth, 1, self.c)


        self._past_size = lightcone_size(self.past_depth, self.c)
        # if self.future_shape == 'flc':
        #     self._future_size = lightcone_size(self.future_depth, self.c) - 1
        # else:
        #     self._future_size = V_size(self.past_depth, self.future_depth, self.c)
        self._future_size = len(self._base_future_t)


    def scan_data(self, data, past_only=False):
        '''
        Deprecated, use gather instead.
        '''
        self.plcs = []
        self.futures = []
        self.data = data
        T, X = np.shape(data)
        # Amount of padding needed to handle periodic boundary conditions in space
        # so that lightcones at the edge of the data can be gathered
        padding = int((self._block_width - 1) / 2)
        padded_data = np.pad(data, ((0,0),(padding, padding)), 'wrap')

        clipped_T = T - self.past_depth - self.future_depth
        base_anchor = (self.past_depth, padding)
        base_plc_t, base_plc_x = plc_gen(base_anchor, self.past_depth, self.c)
        if self.future_shape == 'flc':
            base_future_t, base_future_x = flc_gen(base_anchor, self.future_depth, self.c)
        else:
            base_future_t, base_future_x = V_f_gen(base_anchor, self.past_depth, self.future_depth, self.c)

        for t,x in product(range(clipped_T), range(X)):
            plc_extractor_t = base_plc_t + t
            plc_extractor_x = base_plc_x + x
            plc = tuple(padded_data[plc_extractor_t, plc_extractor_x])
            self.plcs.append(plc)

            if not past_only:
                future_extractor_t = base_future_t + t
                future_extractor_x = base_future_x + x
                future = tuple(padded_data[future_extractor_t, future_extractor_x])
                self.futures.append(future)


    def gather(self, data, past_only=False):
        '''
        Gathers pasts and futures as numpy arrays.
        If it works well, should replace .scan() / .scan_data()
        and .plc_field() / .future_field() should be updated accordingly,
        i.e. just remove the first step of converting to np.arrays, since they
        already will be. And use self.pasts instead of self.plcs
        '''
        self.data = data
        # Cut out the time margin
        T, X = np.shape(data)
        # Amount of padding needed to handle periodic boundary conditions in space
        # so that lightcones at the edge of the data can be gathered
        padded_data = np.pad(data, ((0,0),(self._padding, self._padding)), 'wrap')
        clipped_T = T - self.past_depth - self.future_depth

        # Instantiate the arrays to store the gathered pasts and futures.
        self.plcs = np.zeros((clipped_T*X, self._past_size),
                                dtype=data.dtype)

        self.futures = np.zeros((clipped_T*X, self._future_size), dtype=data.dtype)

        # Search through the field and gather all possible pasts and futures.
        i = 0
        for t, x in product(range(clipped_T), range(X)):
            plc_extractor_t = self._base_plc_t + t
            plc_extractor_x = self._base_plc_x + x
            self.plcs[i] = padded_data[plc_extractor_t, plc_extractor_x]

            if not past_only:
                future_extractor_t = self._base_future_t + t
                future_extractor_x = self._base_future_x + x
                self.futures[i] = padded_data[future_extractor_t, future_extractor_x]
            i += 1

    def gather_transitions(self, data):
        '''

        '''
        # Cut out the time margin
        T, X = np.shape(data)
        # Amount of padding needed to handle periodic boundary conditions in space
        # so that lightcones at the edge of the data can be gathered
        padded_data = np.pad(data, ((0,0),(self._padding+self.c, self._padding+self.c)), 'wrap')
        clipped_T = T - self.past_depth - self.future_depth

        from_plcs = []
        fringes = []
        to_plcs = []

        # Note: since the field here has extra spatial padding for gathering transitions,
        # have to shift all base templates x coordinates by that extra padding, which is c.
        for t,x in product(range(clipped_T), range(X)):
            # right transition
            from_plc_t = self._base_plc_t + t
            from_plc_x = self._base_plc_x + self.c + x
            from_plc = tuple(padded_data[from_plc_t, from_plc_x])
            from_plcs.append(from_plc)
            to_plc_t = self._base_plc_t + t
            to_plc_x = self._base_plc_x + self.c + (x+1)
            to_plc = padded_data[to_plc_t, to_plc_x]
            to_plcs.append(tuple(to_plc))
            right_fringe_t = self._right_fringe_t + t
            right_fringe_x = self._right_fringe_x + self.c + x
            right_fringe = padded_data[right_fringe_t, right_fringe_x]
            fringes.append('r:{}'.format(right_fringe))

            # left transition
            from_plcs.append(from_plc)
            to_plc_t = self._base_plc_t + t
            to_plc_x = self._base_plc_x + self.c + (x-1)
            to_plc = padded_data[to_plc_t, to_plc_x]
            to_plcs.append(tuple(to_plc))
            left_fringe_t = self._left_fringe_t + t
            left_fringe_x = self._left_fringe_x + self.c + x
            left_fringe = padded_data[left_fringe_t, left_fringe_x]
            fringes.append('l:{}'.format(left_fringe))

            # forward transition
            from_plcs.append(from_plc)
            to_plc_t = self._base_plc_t + (t+1)
            to_plc_x = self._base_plc_x + self.c + x
            to_plc = padded_data[to_plc_t, to_plc_x]
            to_plcs.append(tuple(to_plc))
            forward_fringe_t = self._forward_fringe_t + t
            forward_fringe_x = self._forward_fringe_x + self.c + x
            forward_fringe = padded_data[forward_fringe_t, forward_fringe_x]
            fringes.append('t:{}'.format(forward_fringe))

        return (from_plcs, fringes, to_plcs)


    def extract_topological(self, data):
        '''
        Modified version of extract(), hopefully faster with same memory efficiency
        '''
        #self.morph_map = {}
        self.morph_map = defaultdict(set)

        T, X = np.shape(data)
        padded_data = np.pad(data, ((0,0),(self._padding, self._padding)), 'wrap')
        # Cut out the time margin
        clipped_T = T - self.past_depth - self.future_depth
        # Move the templates through the field (excluding the margin)
        for t,x in product(range(clipped_T), range(X)):
            plc_extractor_t = self._base_plc_t + t
            plc_extractor_x = self._base_plc_x + x

            future_extractor_t = self._base_future_t + t
            future_extractor_x = self._base_future_x + x
            # Update the morph map with the extracted lightcones
            plc = tuple(padded_data[plc_extractor_t, plc_extractor_x])
            future = tuple(padded_data[future_extractor_t, future_extractor_x])
            #update_topological_morph(self.morph_map, plc, future)
            self.morph_map[plc].add(future)

    def extract_probabilistic(self, data):
        '''
        Slower, but much more memory efficient version of .extract_morphs()
        '''
        self.morph_map = {}
        # Cut out the time margin
        T, X = np.shape(data)

        padded_data = np.pad(data, ((0,0),(self._padding, self._padding)), 'wrap')

        clipped_T = T - self.past_depth - self.future_depth

        for t,x in product(range(clipped_T), range(X)):
            plc_extractor_t = self._base_plc_t + t
            plc_extractor_x = self._base_plc_x + x

            future_extractor_t = self._base_future_t + t
            future_extractor_x = self._base_future_x + x
            # Update the morph map with the extracted lightcones
            plc = tuple(padded_data[plc_extractor_t, plc_extractor_x])
            future = tuple(padded_data[future_extractor_t, future_extractor_x])
            update_probabilistic_morph(self.morph_map, plc, future)

    def plc_field(self, padded=True):
        '''
        Returns field of all plc labels extracted from the scanned data. Padded
        with 0s where there is not enough horizon for the plc or future (V or flc)
        (i.e. where there is no extraction block).
        '''
        #plc_arrays = np.array(self.plcs)
        plc_arrays = self.plcs

        y, x = np.shape(self.data)
        height = y - self.past_depth - self.future_depth
        reshaped = plc_arrays.reshape(height, x, lightcone_size(self.past_depth, self.c))
        if padded:
            return np.pad(reshaped, ((self.past_depth, self.future_depth), (0,0), (0,0))
                          ,'constant')
        else:
            return reshaped


    def future_field(self, padded=True):
        '''
        Returns field of all flc labels extracted from the scanned data. Padded
        with 0s where there is not enough horizon for the plc or future (V or flc)
        (i.e. where there is no extraction block).
        '''
        #future_arrays = np.array(self.futures)
        future_arrays = self.futures

        y, x = np.shape(self.data)
        height = y - self.past_depth - self.future_depth
        # if self.future_shape == 'flc':
        #     size = lightcone_size(self.future_depth, self.c) - 1
        # else:
        #     size = V_size(self.past_depth, self.future_depth, self.c)
        reshaped = future_arrays.reshape(height, x, self._future_size)
        if padded:
            return np.pad(reshaped, ((self.past_depth, self.future_depth), (0,0), (0,0)),
                          'constant')
        else:
            return reshaped

###################################################################################################

@njit
def grab_lightcones(field, t, x, past_depth, future_depth, c):
    '''
    Since numba does not currently support Python dicts, this is going to be used
    for a hybrid code which extracts one set of lightcones at a time, at spacetime
    point (t,x), rather than grabbing them all at once. This way numba can also handle
    converting them to tuples and I can update the morph dictionary in the same pass as
    through the spacetime field as lightcone extraction.
    '''
    # initialize plc and flc arrays
    plc = []
    flc = []
    # loop through data field to extract past lightcone at point (t,x)
    p = 0
    for d in range(past_depth + 1):
        window_size = 2*d*c + 1
        for w in range(window_size):
            a = -d*c + w
            plc.append(field[t-d, x+a])
            p += 1
    # loop through data field to extract future lightcone at point (t,x)
    f = 0
    for depth in range(future_depth):
        d = depth + 1
        window_size = 2*d*c + 1
        for w in range(window_size):
            a = -d*c + w
            flc.append(field[t+d, x+a])
            f += 1

    return (plc, flc)


def extract_topological_morphs(data_field, past_depth, future_depth, propagation_speed,
                                past_size, future_size):
    '''
    Wrapper function which takes in a spacetime field, calls the grab_lightcones numba
    backend to get lightcone pairs from the field, then updates a morph_map with those
    pairs.
    '''
    max_depth = max(past_depth, future_depth)
    block_width = 2*max_depth*propagation_speed + 1
    padding = int((block_width - 1) / 2)
    T, X = np.shape(data_field)
    clipped_T = T - past_depth - future_depth
    base_t, base_x = (past_depth, padding)
    # everything above here can be pre-computed, except maybe (T,X), when gathering
    # from an ensemble of same-shaped fields.
    padded_data = np.pad(data_field, ((0,0), (padding, padding)), 'wrap')
    #morph_map = {}
    morph_map = defaultdict(set)
    # Move through the field (excluding the margin), grab lightcones and put into morph_map
    for t,x in product(range(clipped_T), range(X)):
        plc, flc = grab_lightcone(padded_data, base_t+t, base_x+x,
                                past_depth, future_depth, propagation_speed,
                                past_size, future_size)
        #update_topological_morph(morph_map, tuple(plc), tuple(flc))
        morph_map[tuple(plc)].add(tuple(flc))
    return morph_map


@jit(nopython=True)
def extract_lightcones(padded_data, T, X, past_depth, future_depth, past_size, future_size,
                    padding, c, base_anchor):
    '''
    Distill the most costly computation in extracting lightcones into a form where
    numba nopython can make full optimization
    '''
    dtype = padded_data.dtype
    base_t, base_x = base_anchor
    clipped_T = T - past_depth - future_depth
    #inds = np.arange(clipped_T*X).reshape((clipped_T, X))
    plcs = np.zeros((clipped_T*X, past_size), dtype=dtype)
    futures = np.zeros((clipped_T*X, future_size), dtype=dtype)
    i = 0
    for t in range(clipped_T):
        for x in range(X):
            # plc_extractor_t = base_plc_t + t
            # plc_extractor_x = base_plc_x + x
            # extracted_plc = padded_data[plc_extractor_t, plc_extractor_x]
            #extracted_plc = np.zeros(past_size, dtype=dtype)
            p = 0
            for d in range(past_depth + 1):
                window_size = 2*d*c + 1
                for w in range(window_size):
                    a = -d*c + w
                    #extracted_plc[p] = padded_data[base_t+t-d, base_x+x+a]
                    plcs[i,p] = padded_data[base_t+t-d, base_x+x+a]
                    p += 1
            #i = inds[t,x]
            #plcs[i] = extracted_plc

            # future_extractor_t = base_future_t + t
            # future_extractor_x = base_future_x + x
            # futures[i] = padded_data[future_extractor_t, future_extractor_x]
            #extracted_flc = np.zeros(future_size, dtype=dtype)
            f = 0
            for depth in range(future_depth):
                d = depth + 1
                window_size = 2*d*c + 1
                for w in range(window_size):
                    a = -d*c + w
                    #extracted_flc[f] = padded_data[base_t+t+d, base_x+x+a]
                    futures[i,f] = padded_data[base_t+t+d, base_x+x+a]
                    f += 1
            #futures[i] = extracted_flc
            i += 1

    return (plcs, futures)


def collect_topological_morphs(plcs, futures):
    '''
    Takes extracted past lightcone - future lightcone pairs and collates them
    into a morph map dictionary.

    Parameters
    ----------
    plcs: nd array
        Array of past lightcone arrays.
    futures: nd array
        Array of associated future arrays.

    Returns
    -------
    morph_map: dict
        Morph map dictionary. Past lightcone configuration tuples are keys
        and their associated topological morphs, given as sets of future lightcone
        array tuples, are vals.
    '''
    morph_map = defaultdict(set)
    # convert lightcone nd arrays to lists of tuples
    plc_tuples = zip(*plcs.T)
    futures_tuples = zip(*futures.T)

    for plc, future in zip(plc_tuples, futures_tuples):
        morph_map[plc].add(future)

    return morph_map
