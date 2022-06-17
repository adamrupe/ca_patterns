'''
Support code to be used by topological and probabilistic modules. Most of this
code is used (or can be used) by both modules, but not all.

author: Adam Rupe
email: atrupe@ucdavis.edu
liscense: BSD
'''

import re
import types
from itertools import product
from collections import defaultdict
import numpy as np

from .CAs import lookup_table

###################################################################################################

def domain_18_configs(length):
    '''
    Produces list of configurations of given length (as np arrays) that are in the
    rule 18 domain shift space. Note this is not taking boundary conditions into
    account (can think of these as sub-configurations).

    *** This is starting with full shift and eliminating configs with forbidden
    words, but this subshift is strictly sofic. Regular expression used to handle this***
    '''
    configs = []
    scan = (0,1)
    # Regular expression for forbidden words that define the domain subshift
    forbidden = '1(00)*1'

    # Generate all binary strings of given length
    for candidate in product(scan, repeat=length):
        st = ''.join(map(str,candidate))
        # Check if the string contains a forbidden word
        m = re.search(forbidden, st)
        # If not, add it to the list as np array
        if m is None:
            configs.append(np.array(candidate))
    return configs

###################################################################################################

def binary_string_gen(length):
    '''
    Generates all binary strings of the given length in lexicographical order.
    Binary strings given as numpy arrays.
    '''
    scan = (0,1)
    # Generate all binary strings of given length
    for string in product(scan, repeat=length):
        yield np.array(string)

###################################################################################################

def convert_to_label(config, A):
    '''
    Returns the integer label for the given configuration array
    '''
    return np.sum(A**np.arange(len(config)) * config[::-1])


def base_array(A, length):
    '''
    Returns part of convert_to_label that can be pre-computed
    when converting multiple configurations of the same length
    with the same alphabet.
    '''
    return A**np.arange(length)


def fast_to_label(config, base_array):
    '''
    Returns the integer label for the given configuration array.

    Faster version of convert_to_label when converting multiple
    configurations with the same length and alphabet.
    '''
    return np.sum(base_array * config[::-1])

###################################################################################################

def convert_to_config(label, size, A, as_array=False):
    '''
    Returns the configuration of the given label. Can be given as string or array.
    '''
    first = np.base_repr(label, A)
    padding = size - len(first)
    final = np.base_repr(label,A,padding)
    if as_array:
        return np.fromstring(final, dtype=np.int8) - 48
    else:
        return final

###################################################################################################

def configuration_to_string(config):
    '''
    Returns given configuration array as string
    '''
    return ''.join(map(str, config))

###################################################################################################

def source_string_length(past_depth, future_depth, propagation_speed, future='flc'):
    '''
    Returns the length of the so called "source string", which determines the past-future
    lightcone template for deterministic 1+1 D systems. For future lightcones, this is the
    length of the base of the patch past lightcone of the base of the future lightcone.
    For full Vs, this is the length of the base of the V + past lightcone.
    '''

    if future not in ['flc', 'V']:
        raise ValueError("future must be either 'flc' or 'V'")

    c = propagation_speed

    if future == 'flc':
        source_length = 4*c*future_depth + 2*c*past_depth + 1

    else:
        source_length = 2*c*future_depth + 2*c*past_depth + 1

    return source_length

###################################################################################################

def compress_dictionary(dictionary):
    comp_keys = []
    comp_vals = []
    for key, val in dictionary.items():
        comp_keys.append(str(key))
        comp_vals.append(str(val))
    return (comp_keys, comp_vals)

def uncompress_dictionary(compressed_dict, topological=True):
    if topological:
        reconstructed = defaultdict(set)
    else:
        reconstructed = {}
    comp_keys, comp_vals = compressed_dict
    for key, val in zip(comp_keys, comp_vals):
        reconstructed.update({eval(key) : eval(val)})
    return reconstructed

###################################################################################################

def save_dict(dictionary, name):
    compressed_dict = compress_dictionary(dictionary)
    with open(name, "wb") as output:
        pickle.dump(compressed_dict, output, protocol=2)

def load_dict(name):
    with open(name, "rb") as input_file:
        compressed_dict = pickle.load(input_file)
    return uncompress_dictionary(compressed_dict)

###################################################################################################

class PlcMachine(object):
    '''

    '''

    def __init__(self, past_depth, propagation_speed):
        '''

        '''
        self.past_depth = past_depth
        self.c = propagation_speed

        self.extractor = Extractor(self.past_depth, 1, self.c, 'flc')

        self.main_map = {}

    def scan_data(self, input_field):
        '''
        The reconstructor grabs statistics (past lightcones, fringes, Vs, etc)
        from these input fields and uses them to create / update the main lists.
        Can use this method to add more data if the statistics have not yet converged.

        Parameters
        ----------
        input_fields - tuple or list of ndarrays, or single ndarray
            Tuple of spacetime field ndarrays. For each spacetime ndarray,
            time should be the vertical and space horizontal.

            May also be a single spacetime field (either in a list / tuple or not)
        '''

        # Extract past lightcones
        self.extractor.scan(input_field)
        # Go through list of extracted plcs, and map unique plcs to arbitrary integer label
        label = 0
        for plc in self.extractor.plcs:
            if plc not in self.main_map:
                self.main_map.update({plc : label})
                label += 1

        # Get plc field without the margin
        plc_field = self.extractor.plc_field(padded=False)
        # Get dimensions of the field without the margin
        time, space = np.shape(plc_field)[:2]
        # Initialize space state field with those dimensions
        self.state_field = np.zeros((time, space), dtype=int)

        # Scan through the plc field and use the state map to fill out the state field
        for t in range(time):
            for s in range(space):
                self.state_field[t,s] = self.main_map[tuple(plc_field[t,s])]

        # Go back and re-pad state field with margin so it is the same shape as the original data
        self.state_field = np.pad(self.state_field, ((self.past_depth, 0), (0,0)),
                                'constant')

    def complexity_field(self, transient=0):
        '''
        Run 'naive' algorithm, just counting the margin as a state (index 0).
        A better version would be to use .extractor.margin_padding to cut out the
        margin, calculate complexity, then pad back in the margin at the end
        with margin complexity 0.
        '''

        state_field = self.state_field[transient+self.past_depth:]

        time, space = np.shape(state_field)
        complexity = np.zeros((time, space), dtype=int)
        states, counts = np.unique(state_field, False, False, True)
        hist = np.zeros(max(states) + 1)
        for state, count in zip(states,counts):
            hist[state] = count
        state_dist = hist / np.sum(hist)
        ent = np.copy(state_dist)
        for i, x in enumerate(ent):
            if x == 0:
                pass
            else:
                ent[i] = -np.log(x)/np.log(2)
        # Filter state field with local complexity map to get complexity field
        complexity = ent[state_field]

        #pad back the transient time cut off for complexity calculation
        return np.pad(complexity, ((transient+self.past_depth, 0), (0, 0)), 'constant')

###################################################################################################

def right_transition_violations(ECA_rule_number, transitions):
    '''
    Quick and dirty by-hand code for ECA lookup table violations in 
    length 3 right-transition fringe words by machines of past depth 3, 
    i.e. right-transition fringes are assumed to be length 4. 
    '''
    lookup = lookup_table(ECA_rule_number, 2, 1)
    violating_words = set()

    for first_transition in transitions:
        first_from, first_fringe, first_to = first_transition
        first = np.array([first_fringe[3], 
                         first_fringe[5], 
                         first_fringe[7], 
                         first_fringe[9]]).astype(int)

        for second_transition in transitions:
            second_from, second_fringe, second_to = second_transition
            if second_from is not first_to:
                pass
            else:
                second = np.array([second_fringe[3], 
                                   second_fringe[5], 
                                   second_fringe[7], 
                                   second_fringe[9]]).astype(int)

                for third_transition in transitions:
                    third_from, third_fringe, third_to = third_transition
                    if third_from is not second_to:
                        pass
                    else:
                        third = np.array([third_fringe[3], 
                                   third_fringe[5], 
                                   third_fringe[7], 
                                   third_fringe[9]]).astype(int)

                        word = np.concatenate((first, second, third))
                        neighborhood_1 = (word[3], word[7], word[11])
                        neighborhood_2 = (word[2], word[6], word[10])
                        neighborhood_3 = (word[1], word[5], word[9])
                        if (lookup[neighborhood_1] != word[10]) or \
                           (lookup[neighborhood_2] != word[9]) or \
                            (lookup[neighborhood_3] != word[8]):
                                violating_words.add(tuple(word))
                        else:
                            pass

    return violating_words

def left_transition_violations(ECA_rule_number, transitions):
    '''
    Quick and dirty by-hand code for ECA lookup table violations in 
    length 3 right-transition fringe words by machines of past depth 3, 
    i.e. right-transition fringes are assumed to be length 4. 
    '''
    lookup = lookup_table(ECA_rule_number, 2, 1)
    violating_words = set()
    
    for first_transition in transitions:
        first_from, first_fringe, first_to = first_transition
        first = np.array([first_fringe[3], 
                         first_fringe[5], 
                         first_fringe[7], 
                         first_fringe[9]]).astype(int)
        
        for second_transition in transitions:
            second_from, second_fringe, second_to = second_transition
            if second_from is not first_to:
                pass
            else:
                second = np.array([second_fringe[3], 
                                   second_fringe[5], 
                                   second_fringe[7], 
                                   second_fringe[9]]).astype(int)
                
                for third_transition in transitions:
                    third_from, third_fringe, third_to = third_transition
                    if third_from is not second_to:
                        pass
                    else:
                        third = np.array([third_fringe[3], 
                                   third_fringe[5], 
                                   third_fringe[7], 
                                   third_fringe[9]]).astype(int)
                        
                        word = np.concatenate((first, second, third))
                        neighborhood_1 = (word[11], word[7], word[3])
                        neighborhood_2 = (word[10], word[6], word[2])
                        neighborhood_3 = (word[9], word[5], word[1])
                        if (lookup[neighborhood_1] != word[10]) or \
                           (lookup[neighborhood_2] != word[9]) or \
                            (lookup[neighborhood_3] != word[8]):
                                violating_words.add(tuple(word))
                        else:
                            pass
        
    return violating_words

def forward_transition_violations(ECA_rule_number, transitions):
    '''
    Quick and dirty by-hand code for ECA lookup table violations in 
    length 3 right-transition fringe words by machines of past depth 3, 
    i.e. right-transition fringes are assumed to be length 4. 
    '''
    lookup = lookup_table(ECA_rule_number, 2, 1)
    violating_words = set()
    
    for first_transition in transitions:
        first_from, first_fringe, first_to = first_transition
        first = np.array([first_fringe[3], 
                         first_fringe[5], 
                         first_fringe[7], 
                         first_fringe[9],
                         first_fringe[11],
                         first_fringe[13],
                         first_fringe[15]]).astype(int)
        
        for second_transition in transitions:
            second_from, second_fringe, second_to = second_transition
            if second_from is not first_to:
                pass
            else:
                second = np.array([second_fringe[3], 
                                   second_fringe[5], 
                                   second_fringe[7], 
                                   second_fringe[9],
                                   second_fringe[11],
                                   second_fringe[13], 
                                   second_fringe[15]]).astype(int)
                
                for third_transition in transitions:
                    third_from, third_fringe, third_to = third_transition
                    if third_from is not second_to:
                        pass
                    else:
                        third = np.array([third_fringe[3], 
                                   third_fringe[5], 
                                   third_fringe[7], 
                                   third_fringe[9],
                                   third_fringe[11],
                                   third_fringe[13],
                                   third_fringe[15]]).astype(int)
                        
                        word = np.concatenate((first, second, third))
                        neighborhood_1 = (word[19], word[10], word[1])
                        neighborhood_2 = (word[2], word[11], word[20])
                        neighborhood_3 = (word[17], word[8], word[0])
                        neighborhood_4 = (word[8], word[0], word[9])
                        neighborhood_5 = (word[0], word[9], word[18])
                        neighborhood_6 = (word[15], word[7], word[16])
                        if (lookup[neighborhood_1] != word[17]) or \
                           (lookup[neighborhood_2] != word[18]) or \
                            (lookup[neighborhood_3] != word[15]) or \
                            (lookup[neighborhood_4] != word[7]) or \
                            (lookup[neighborhood_5] != word[16]) or \
                            (lookup[neighborhood_6] != word[14]):
                                violating_words.add(tuple(word))
                        else:
                            pass
        
    return violating_words