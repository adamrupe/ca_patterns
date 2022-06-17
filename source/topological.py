'''
Contains code to handle reconstruction of topological spacetime machines from
data. Statistics can be gathered from an ensemble of fields, and the causal
equivalence relation can be constructed from futureions of flcs or Vs.

Main flow for reconstructing a topological machine from data is as follows:
1) initialize TopologicalReconstructor object with input parameters of past
lightcone depth, future lightcone depth, and propagation speed.
2) Generate or otherwise provide a training data set to run inference on and run
the scan_data method of the TopologicalReconstructor on this data. The scan_data
method uses an Extractor object and uses its extract method to create and populate
a morph_map. Data parallelization can be done, utilizing the merge method of
the TopologicalReconstructor class to merge the morph_maps of multiple Reconstructors.
3) Can use the plot_statistics method of the TopologicalReconstructor to check
whether the number of unique past lightcone - future lightcone pairs has reached an
asymptote and thus have reached topological convergence. If it has not converged, can
simply run more data through the scan_data method of the Reconstructor (or merge
more reconstructors run in parallel).
4) If the morph_map has converged to a satisfactory degree can run the
reconstruct_states method of the TopologicalReconstructor. This initializes a
TopologicalMachine object with the same parameters used to initialize the Reconstructor
and then runs the construct_states method of the TopologicalMachine using the
morph_map inferred by the Reconstructor. This TopologicalMachine is now an
attribute of the Reconstructor.
5) Can grab the TopologicalMachine from the TopologicalReconstructor and use it
for further analysis, including filtering a representative spacetime field.


author: Adam Rupe
email: atrupe@ucdavis.edu
liscense: BSD
'''
import types
import time
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

from .lightcones import *
from .utilities import *
from .CAs import lookup_table, numba_spacetime

###################################################################################################

class TopologicalState(object):
    '''
    Topological state object. Centered around the topological morph which defines
    the state, as well as the past lightcones which have this morph and thus
    belong to the state. Also always carries a simple integer label to identify
    the state, with an optional more descriptive string label.

    Attributes
    ----------
    index: int
        Integer label for the state.

    label: str
        Arbitrary string label for the state.

    plcs: set
        Set of configuration vector tuples for the past lightcones that belong to the state.

    morph: set
        Set of configuration vectors (given as tuples) for future lightcones or
        Vs that belong to the state's morph. This morph defines the topological state.
    '''

    def __init__(self, state_index, first_plc, morph, label=None):
        '''
        Initializes the TopologicalState state class with the given input parameters.
        This init protocol is designed for use with hierarchical agglomerative
        clustering where a state is initialized with a single past lightcone.

        Parameters
        ----------
        state_index: int
            Simple integer label which idententifies the state.

        first_plc: array
            Initial past lightcone which first populates the state.

        morph: dict
            Topological morph carried by the initializing past lightcone which
            defines the topological state.

        label: str, optional (default=None)
            Optional string for a more detailed label for the state.

        '''
        # Integer index label of the state
        self.index = state_index
        # Arbitrary label for the state
        self.label = label
        # The state's future morph (this is what defines the state)
        self.morph = morph
        # Set of pasts in the state (equivalance class of past lightcones)
        self.plcs = {first_plc}

    def contracted_plcs(self, original_depth, desired_depth, c):
        '''
        Takes all past lightcones in the state's equivalence class and
        contracts them to the desired depth. Returns the contracted past
        lightcones as a set of past lightcones with the desired depth.
        '''
        contracted_plcs = set()
        for plc in self.plcs:
            contracted_plc = contract_plc(plc,
                                        original_depth,
                                        desired_depth,
                                        c)
            contracted_plcs.add(tuple(contracted_plc))
        return contracted_plcs

    def contracted_morph(self, original_depth, desired_depth, c):
        '''
        Takes all future lightcones in the original morph and contracts them
        to the given desired depth. Returns the contracted morph as a set
        of future lightcones with desired depths.
        '''
        contracted_morph = set()
        for flc in self.morph:
            contracted_flc = contract_flc(flc,
                                        original_depth,
                                        desired_depth,
                                        c)
            contracted_morph.add(tuple(contracted_flc))
        return contracted_morph

###################################################################################################

def contracted_decontamination(domain_states,
                               non_domain_states,
                               original_plc,
                               contracted_plc,
                               original_flc,
                               contracted_flc,
                               c):
    '''
    For stocahstic domain CAs the local causal states can have difficulty in
    precisely delineating the boundary where domain ends and non-domain begins,
    specifically coherent structures. This is due to "contamination", where the
    structure runs through either the past lightcones, future lightcones, or both,
    of what should be a domain local causal state. Contracted decontamination is
    an algorithm to correct for this by identifying states which fit domain criteria
    after contracting the past lightcones, future lightcones, or both, of that state.
    The criteria when contracting past lightcones is that, after contraction, all of
    the past lightcones of the state are contracted past lightcones of domain states.
    The criteria when contracting future lightcones is that, after contraction, the
    contracted morph of the state CONTAINS (but does not necessarily equal) the
    contracted morph of one of the domain states. This, of course, requires knowing
    what the pure domain states, without the presence of contaminating structures.
    The future lightcone part limits this algorithm to topological reconstruction
    only.

    *** TO DO ***
    Will it still work fine if non_domain_states contains the domain_states, which
    happens with hierarchical reconstruction?

    Parameters
    ----------

    Returns
    -------
    '''
    decontaminated_states = set()
    if contracted_plc > original_plc:
        raise ValueError("Contracted plc depth must be less than or equal to original depth.")
    # If original = contracted, do nothing
    elif contracted_plc < original_plc:
        # Get set of all domain past lightcones
        domain_plcs = set().union(*[domain.contracted_plcs(original_plc, contracted_plc, c) \
                                    for domain in domain_states])
        # Mark states as decontaminated if, after plc contraction, they contain only domain plcs
        for non_domain in non_domain_states:
            if non_domain.contracted_plcs(original_plc, contracted_plc, c) <= domain_plcs:
            # if len(non_domain.contracted_plcs(original_plc, contracted_plc, c) \
            #         - domain_plcs) == 0:
                decontaminated_states.add(non_domain.index)
    # Mark states as decontaminated if, after morph contraction, the contain a domain morph
    if contracted_flc > original_flc:
        raise ValueError("Contracted flc depth must be less than or equal to original depth.")
    # If original = contracted, do nothing
    elif contracted_flc < original_flc:
        for non_domain in non_domain_states:
            for domain in domain_states:
                if domain.contracted_morph(original_flc, contracted_flc, c) \
                <= non_domain.contracted_morph(original_flc, contracted_flc, c):
                    decontaminated_states.add(non_domain.index)
    return decontaminated_states

###################################################################################################

class TopologicalMachine(object):
    '''
    Topological machine class, consists mainly of a set of topological states
    as well as simple (topological) transitions among these states.

    ***Potential issue that would be nice to fix: machine is initialized with a
    given input of past_depth, future_depth, and propagation_speed. However,
    the states are actually constructed with the construct_states method which
    takes a morph_map as input, and there could be a mismatch between the past
    and future lightcones in this morph map and the init parameters. Perhaps should
    initialize with just the morph_map as input (and perhaps c as well), and could
    somehow extract the past and future depths from that.

    With my current code architecture, this object is typically initialized and
    construct_states method run within the TopologicalReconstructor object, so the
    above issue isn't a problem there. But still this method does exist on its own
    and should still think about fixing this, or at least raise an exception. ***

    Attributes
    ----------
    past_depth: int
        Finite horizon depth of past lightcones chosen for inference of this
        topological machine.

    future_depth: int
        Finite horizon depth of future lightcones chosen for inference of this
        topological machine.

    c: int
        Finite speed of interaction / perturbation propagation used for inference.
        Either explicitly specified by the system (like with cellular automata) or
        chosen as an inference parameter to capture specific physics (e.g. chosing
        advection scale rather than accoustic scale for climate).

    extractor: Exctractor class
        Extractor object used in the filter_data method to extract past lightcones
        from the input data to be filtered. Only need past lightcones for this,
        which are inputs for the epsilon-map, therefore the future_depth of this
        Extractor is just set to 1.

    state_index: int
        Integer which is iterated over and incrimented during state
        reconstruction to provide unique integer labels for the states. Initilized
        at '1' since the '0' label is reserved for the margins (should also be
        used for past lightcones which are present in data to be filtered but was
        not present in data inferred over and thus don't belong to any of the
        topological states).

    states: list
        List of the topological states of the machine which have been inferred
        from the given morph map in the construct_states method.

    present_states: list
        List of topological states which are present in the input field of the
        filter_data method. Initialized as None, populated only after running
        filter_data method.

    state_field: array
        2-D spacetime array, shape=(time, space), of causal state field produced
        after filtering input data of the filter_data method. Initialized as None,
        array created only after running filter_data method. Array will have same
        shape as the input field given for filter_data method so that coordinates
        in each match. Margins in the state_field are given value '0'.

    epsilon_map: dict
        Epsilon function which maps past lightcones to topological states, given
        as a dictionary with past lightcone arrays (given as tuples) as keys and
        TopologicalState objects as vals. Populated during the construct_states
        method.

    transition attributes to be filled out after the code for this is decided upon.
    '''

    def __init__(self, past_depth, future_depth, propagation_speed):
        '''
        Initializes the TopologicalMachine object with the given parameters.

        Parameters
        ----------
        past_depth: int
            Finite horizon depth of past lightcones chosen for inference of this
            topological machine.

        future_depth: int
            Finite horizon depth of future lightcones chosen for inference of this
            topological machine.

        propagation_speed: int
            Finite speed of interaction / perturbation propagation used for inference.
            Either explicitly specified by the system (like with cellular automata) or
            chosen as an inference parameter to capture specific physics (e.g. chosing
            advection scale rather than accoustic scale for climate).


        '''
        self.past_depth = past_depth
        self.future_depth = future_depth
        self.c = propagation_speed
        # future depth set to 0 for no future margin while filtering
        self.extractor = Extractor(self.past_depth, 0,
                                    self.c)

        self._state_index = 1
        self.states = []
        self.present_states = None
        self.state_field = None
        self.epsilon_map = {}
        self.transitions = {'t': set(), 'r': set(), 'l': set()}

    def construct_states(self, morph_map):
        '''
        Performs state clustering algorithm on the given morph map of unique past
        lightcones and their associated morphs. This is a hierarchical agglomerative
        clustering algorithm.

        Unique past lightcones and their morphs can be gathered from data using
        TopologicalReconstructor class.

        Parameters
        ----------
        morph_map: dict
            Dictionary which has unique past lightcone arrays (given as tuples) as
            keys and their associated morphs, given as sets of future lightcone
            arrays (also given as tuples), as vals.
        '''
        for plc, morph in morph_map.items():
            # For each plc in the morph map, check to see if its morph matches the morph of
            # any current state (plc cluster)
            for state in self.states:
                # If they match, add plc to that state, and update the epsilon map
                if morph == state.morph:
                    state.plcs.add(plc)
                    self.epsilon_map.update({plc : state})
                    break

            # If morph does not match any existing state, create new state and update epsilon map
            else:
                new_state = TopologicalState(self._state_index, plc, morph)
                self.states.append(new_state)
                self._state_index += 1
                self.epsilon_map.update({plc : new_state})

    def reset_states(self, new_morph_map):
        '''
        Method for starting over and reconstructing states from scratch with a new
        morph map.

        Parameters
        ----------
        new_morph_map: dict
            Dictionary which has unique past lightcone arrays (given as tuples) as
            keys and their associated morphs, given as sets of future lightcone
            arrays (also given as tuples), as vals.
        '''
        # Reset necessary instance variables
        self._state_index = 1
        self.states = []
        self.present_states = None
        self.state_field = None
        self.epsilon_map = {}
        # Run construct_states after the reset
        self.construct_states(new_morph_map)

    def number_of_states(self):
        '''
        Returns the number of states of the topological machine.

        Returns
        -------
        out: int
            Number of states of the topological machine.
        '''
        return len(self.states)

    def filter_data(self, spacetime_data):
        '''
        Filters the input data with the state_map (epsilon map) and produces
        a topological state field as the attribute self.state_field. The state
        field has the same shape as the input data given here.

        Past lightcones not recognized by the inferred epsilon function will be
        mapped to a "null state" with label 0.

        Parameters
        ----------
        spacetime_data: array
            Numpy array of the spacetime field which is to be filtered. Expected
            shape = (time, space).

        *** To do ***
        - change to nditer
        - get rid of use of Extractor (?)
        '''
        self.present_states = []
        self.extractor.gather(spacetime_data, past_only=True)
        plc_field = self.extractor.plc_field(padded=False)
        time, space = np.shape(plc_field)[:2]
        self.state_field = np.zeros((time, space), dtype=int)

        # Scan through the plc field and use the state map to fill out the state field
        # If a plc is not recognized, map it to "null" state 0
        for t,s in product(range(time), range(space)):
            try:
                state = self.epsilon_map[tuple(plc_field[t,s])]
                self.state_field[t,s] = state.index
            except KeyError:
                self.state_field[t,s] = 0

        # Fill in present_states with states that are present in the filtered data
        present_states = np.unique(self.state_field)
        # Remember that state index 0 is a flag for 'not actually a state'
        for state_label in present_states[present_states != 0]:
            state_index = state_label - 1
            self.present_states.append(self.states[state_index])

        # Go back and re-pad state field with margin so it is the same shape as the original data
        # Note 0 future padding so states can be filled all the way to the end
        # Remember that the extractor future depth has been set to 0, which is why this works!
        self.state_field = np.pad(self.state_field, ((self.past_depth, 0), (0,0)),
                                'constant')

    def complexity_field(self, transient=0):
        '''
        Counts topological states in the field, cutting out a transient if supplied,
        to create a state distribution. Pointwise entropy of this state distribution
        calculated and used to create a local statistical complexity field as output.
        This complexity field will have the same shape as the input field given during
        the filter_data method.

        Parameters
        ----------
        transient: int, optional (default=0)
            Number of time steps (beyond the past lightcone margin) to cut out before
            counting states to create state distribution.
        '''
        total_time = np.shape(self.state_field)[0]
        state_field = self.state_field[transient+self.past_depth : total_time-self.future_depth]

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
        return np.pad(complexity, ((transient+self.past_depth, self.future_depth), (0, 0)), 'constant')

    def number_of_unique_plcs(self):
        '''
        Gives the number of unique past lightcones that were seen during inference
        for reconstruction of the TopologicalMachine. This aggregates the number
        of past lightcones accross all the topological states of the Machine.

        Returns
        -------
        out: int
            Number of unique past lightcones.
        '''
        return np.sum([len(state.plcs) for state in self.states])

    def number_of_unique_pairs(self):
        '''
        Gives the number of unique past lightcone - future lightcone pairs which
        have been seen in all the topological morphs in the morph map used to
        create the topological machine instance.

        Returns
        -------
        out: int
            Number of unique lightcone pairs in the machine instance's morph map.
        '''
        return np.sum([len(state.morph) for state in self.states])

    def stats(self, order_of_magnitude=False):
        '''
        -to be added-

        want to have an extra key to the (topological) morph maps: 'total'
        (or something similar, maybe 'stats' to match) which will have a val equal to the total number
        of lightcone pairs scanned in creating the morph map instance. This is
        needed because the topological morphs don't keep track of total counts,
        just the unique lightcone pairs seen. This will require change of code
        throughout the whole module.
        '''

###################################################################################################

class TopologicalReconstructor(object):
    '''
    Reconstructs a local topological spacetime epsilon machine from given input
    spacetime field(s) using the given estimation paramters.

    Attributes
    ----------
    self.past_depth: int
        The finite past lightcone depth to be used for spacetime machine reconstruction.

    self.future_depth: int
        The finite future lightcone or V depth to be used for spacetime machine
        reconstruction.

    self.c: int
        Speed of information propagation in the spacetime field.

    self.extractor: Extractor
        Extractor class used to gather statistics (lightcones, Vs, fringes) from
        each spacetime field.

    self.main_lists: list, [plcs, flc_morphs, V_morphs, start_plcs, fringes, end_plcs]

        plcs and V_morphs are ordered / arranged such that Vs[i] is the list of
        associated Vs (i.e. the topological morph) that corresponds to the
        unique plc, plcs[i].

        Both types of morphs are given as sets. That is, flc_morphs and V_morphs
        are both lists of sets.

        Likewise the start_plcs, fringes, end_plcs are ordered / arranged such
        that the elements of zip(start_plcs, fringes, end_plcs) should be unique
        and represent potentially unique transitions.

        Eventually should turn into a property for some level of privacy

    self.main_plc_sizes: list
        List of the number of unique plcs in the main list after extraction from
        each spacetime field. Element i is the the number of unique plcs in the
        main list after extraction from the ith field.

        Eventually should turn into a property for some level of privacy

    self.main_flc_sizes: list
        List of the total number of elements in the main list of flc morphs.
        Element i is the total number of elements in the main flc morph list
        after extraction from the ith field.

        Eventually should turn into a property for some level of privacy

    self.main_V_sizes: list
        List of of the total number of elements in the main list of V morphs.
        Element i is the total number of elements in the main V morph list
        after extraction from the ith field.

        Eventually should turn into a property for some level of privacy


    '''

    def __init__(self, past_depth, future_depth, propagation_speed, future='flc',
                 plc_shape='full', flc_shape='full'):
        '''

        '''
        self.past_depth = past_depth
        self.future_depth = future_depth
        self.c = propagation_speed
        self.future = future
        if future not in ['flc', 'V_f', 'V_r', 'V_l', 'V_t']:
            raise ValueError("future_shape must be either 'flc', 'V_f', 'V_r', 'V_l', or 'V_t'")

        self.extractor = Extractor(self.past_depth, self.future_depth, self.c, self.future,
                                    plc_shape=plc_shape, flc_shape=flc_shape)

        self.main_map = defaultdict(set)
        self.main_morph_sizes = []
        self.merge_morph_sizes = []
        self.total_pairs = 0

        self.machine = None
        self.transition_sizes = []

    def scan_data(self, input_fields):
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
        # Check if input is a single field, if so put it in a tuple
        if not isinstance(input_fields, (tuple, list, types.GeneratorType)):
            field_set = (input_fields,)
        else:
            field_set = input_fields
        # Search through the list / tuple of fields
        for field in field_set:
            T,X = np.shape(field)
            self.total_pairs += (T-self.past_depth-self.future_depth) * X
            # For each field, extract the single-field morph map
            self.extractor.extract_topological(field)
            # Use this object to update the main morph map
            merge_topological_morphs(self.main_map, self.extractor.morph_map)

            # Update morph size statistics
            morph_size = np.sum([len(morph) for morph in self.main_map.values()])
            self.main_morph_sizes.append(morph_size)

        # Clear the last merged extractor map
        self.extractor.morph_map = defaultdict(set)

    def fast_scan(self, input_fields):
        '''
        Playing around with numba-boosted pure-numpy lightcone extraction.

        For simplicity, only doing future lightcones. No Vs.
        '''
        # Check if input is a single field, if so put it in a tuple
        if not isinstance(input_fields, (tuple, list, types.GeneratorType)):
            field_set = (input_fields,)
        else:
            field_set = input_fields

        if self.future == 'flc':
            max_depth = max(self.past_depth, self.future_depth)
            block_width = 2*max_depth*self.c + 1
        else:
            raise ValueError("This method only works with future lightcones. Use \
            self.future = 'flc'.")

        padding = int((block_width - 1) / 2)
        # Starting point in top left corner (excluding margin)
        base_anchor = (self.past_depth, padding)

        past_size = lightcone_size(self.past_depth, self.c)
        future_size = lightcone_size(self.future_depth, self.c) - 1

        for field in field_set:
            T, X = field.shape
            padded_data = np.pad(field, ((0,0), (padding, padding)), 'wrap')
            plcs, futures = extract_lightcones(padded_data, T, X,
                                            self.past_depth,
                                            self.future_depth,
                                            past_size,
                                            future_size,
                                            padding,
                                            self.c,
                                            base_anchor)
            # self.extractor.gather(field)
            # plcs, futures = (self.extractor.plcs, self.extractor.futures)
            new_map = collect_topological_morphs(plcs, futures)
            merge_topological_morphs(self.main_map, new_map)

            # Update morph size statistics
            morph_size = np.sum([len(morph) for morph in self.main_map.values()])
            self.main_morph_sizes.append(morph_size)

    def transition_scan(self, input_fields):
        '''
        To be run after an epsilon-map is inferred. Scans data to collect fringe-labelled
        local causal state transitions. Can track the total number of unique transitions
        with transitions_statistics() method to see if transitions have converged.

        '''
        if self.machine is None:
            raise ValueError('There is no machine to gather transitions for.')

        if not isinstance(input_fields, (tuple, list, types.GeneratorType)):
            field_set = (input_fields,)
        else:
            field_set = input_fields

        for field in field_set:
            from_plcs, fringes, to_plcs = self.extractor.gather_transitions(field)
            for fro, fringe, to in zip(from_plcs, fringes, to_plcs):
                from_state = self.machine.epsilon_map[fro].index
                to_state = self.machine.epsilon_map[to].index
                direction = fringe[0]
                self.machine.transitions[direction].add((from_state, fringe, to_state))

            #self.transition_sizes.append(len(self.machine.transitions))
            self.transition_sizes.append(np.sum([len(trans) for trans in self.machine.transitions.values()]))

    def test_scan(self, input_fields):
        '''
        Playing around with numba-boosted pure-numpy lightcone extraction.

        For simplicity, only doing future lightcones. No Vs.
        '''
        # Check if input is a single field, if so put it in a tuple
        if not isinstance(input_fields, (tuple, list, types.GeneratorType)):
            field_set = (input_fields,)
        else:
            field_set = input_fields

        if self.future != 'flc':
            raise ValueError("This method only works with future lightcones. Use \
            self.future = 'flc'.")

        past_size = lightcone_size(self.past_depth, self.c)
        future_size = lightcone_size(self.future_depth, self.c) - 1

        for field in field_set:
            new_map = extract_topological_morphs(field,
                                            self.past_depth,
                                            self.future_depth,
                                            self.c,
                                            past_size,
                                            future_size)

            merge_topological_morphs(self.main_map, new_map)

            # Update morph size statistics
            morph_size = np.sum([len(morph) for morph in self.main_map.values()])
            self.main_morph_sizes.append(morph_size)

        # clear last map from memory
        del new_map


    def merge(self, new_main_map):
        '''
        Merges the main morph map of this Reconstructor object with the input
        main morph map of another Reconstructor. Mainly used to combine results from
        multiple TopologicalReconstructors run in parallel.

        Parameters
        ----------
        new_main_map: dict
            main map of the input Reconstructor object to be merged, i.e.
            NewRconstructor.main_map

        '''
        merge_topological_morphs(self.main_map, new_main_map)

        # Update merged list sizes to track statistical convergence of morphs
        map_ = self.main_map
        morph_size = np.sum([len(map_[plc]) for plc in map_])
        self.merge_morph_sizes.append(morph_size)

    def merge_plot(self, n_fields=None, field_shape=None, size=10):
        '''
        Plots the main morph sizes as a function of the number of TopologicalReconstructors
        merged. Meant to be used by a "dummy" Reconstructor that merges and tracks results from
        a sequecne of other Reconstructors run in parallel.

        Parameters
        ----------
        n_fields: int
            Number of fields handled by each Reconstructor.

        field_shape: array-like (t,x)
            Shape of the spacetime fields scanned by the Reconstructors


        '''
        x = np.arange(1, len(self.merge_morph_sizes)+1)
        if n_fields is not None and field_shape is not None:
            scale = n_fields * np.product(field_shape)
            x *= scale
        plt.figure(figsize = (size, size))
        plt.plot(x, self.merge_morph_sizes, label = '{}'.format(self.future))
        plt.legend()
        plt.show()

    def plot_statistics(self, size=10):
        '''
        Simple matplotlib plot of flc and V morph sizes as a funciton of number
        of spacetime fields scanned.
        '''
        plt.figure(figsize = (size, size))
        plt.plot(self.main_morph_sizes, label = '{}'.format(self.future))
        plt.legend()
        plt.show()

    def transition_statistics(self, size=10):
        '''
        Simple matplotlib plot of number of unique transitions as a function of the
        number of spacetime fields scanned.
        '''
        plt.figure(figsize = (size, size))
        plt.plot(self.transition_sizes)
        plt.show()

    def reconstruct_states(self):
        '''
        Uses the current main lists to reconstruct the states of a
        topological_machine object.
        '''
        self.machine = TopologicalMachine(self.past_depth, self.future_depth,
                                            self.c)
        self.machine.construct_states(self.main_map)

    def reconstruct_transitions(self):
        '''
        Uses the current main lists to reconstruct the transitions of a
        topological_machine object.
        '''

    def reconstruct_machine(self, future):
        '''
        Uses the current main lists to reconstruct the states and transitions of a
        topological_machine object.
        '''

###################################################################################################

class HierarchicalReconstructor(object):
    '''
    Generalized reconstructor for hierarchical reconstruction of local causal states
    for fields with coherent structures on top of non-zero entropy density domains /
    backgrounds.

    Capable of performing state reconstruction in mulitple stages; an epsilon map
    is learned in the initial stage and used to constuct local causal states, and
    at the next stage this map is remembered and modified to incorporate only new
    past lightcones and their topological morphs.

    Also capable of using the epsilon map of the prior stage of reconstruction to
    perform decontamination of the current stage.

    Attributes
    ----------
    self.past_depth: int
        The finite past lightcone depth to be used for spacetime machine reconstruction.

    self.future_depth: int
        The finite future lightcone or V depth to be used for spacetime machine
        reconstruction.

    self.c: int
        Speed of information propagation in the spacetime field.

    self.extractor: Extractor
        Extractor class used to gather statistics (lightcones, Vs, fringes) from
        each spacetime field.
    '''

    def __init__(self, past_depth, future_depth, propagation_speed, future='flc'):
        '''

        '''
        self.past_depth = past_depth
        self.future_depth = future_depth
        self.c = propagation_speed
        self.future = future
        if future not in ['flc', 'V']:
            raise ValueError("future must be either 'flc' or 'V'")

        self.extractor = Extractor(self.past_depth, self.future_depth, self.c, self.future)

        self.morph_maps = [defaultdict(set)]
        self.morph_sizes = [[]]
        #self.merge_morph_sizes = []
        self.scan_time = 0
        self.total_pairs = 0

        self.current_stage = 0
        self.known_plcs = set()
        self.current_map = self.morph_maps[self.current_stage]

        self.machine = None

    def scan_data(self, input_fields):
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
        start = time.clock()
        # Check if input is a single field, if so put it in a tuple
        if not isinstance(input_fields, (tuple, list, types.GeneratorType)):
            field_set = (input_fields,)
        else:
            field_set = input_fields
        # Search through the list / tuple of fields
        for field in field_set:
            # Get the number of lightcone pairs scanned in this field and add it
            # to the total count
            T,X = np.shape(field)
            self.total_pairs += (T-self.past_depth-self.future_depth) * X
            # For each field, extract the single-field morph map
            self.extractor.extract_topological(field)
            # Use this object to update the current morph map, excluding the known plcs
            # from the previous stage(s)
            hierarchical_topological_merge(self.current_map, self.extractor.morph_map,
                            self.known_plcs)

            # Update morph size statistics
            morph_size = np.sum([len(morph) for morph in self.current_map.values()])
            self.morph_sizes[self.current_stage].append(morph_size)

        # Clear the last merged extractor map
        self.extractor.morph_map = defaultdict(set)

        end = time.clock()
        self.scan_time += (end - start) / 60


    def fast_scan(self, input_fields):
        '''
        Playing around with numba-boosted pure-numpy lightcone extraction.

        For simplicity, only doing future lightcones. No Vs.
        '''
        # Check if input is a single field, if so put it in a tuple
        if not isinstance(input_fields, (tuple, list, types.GeneratorType)):
            field_set = (input_fields,)
        else:
            field_set = input_fields

        if self.future == 'flc':
            max_depth = max(self.past_depth, self.future_depth)
            block_width = 2*max_depth*self.c + 1
        else:
            raise ValueError("This method only works with future lightcones. Use \
            self.future = 'flc'.")

        padding = int((block_width - 1) / 2)
        # Starting point in top left corner (excluding margin)
        base_anchor = (self.past_depth, padding)

        past_size = lightcone_size(self.past_depth, self.c)
        future_size = lightcone_size(self.future_depth, self.c) - 1

        for field in field_set:
            T, X = field.shape
            padded_data = np.pad(field, ((0,0), (padding, padding)), 'wrap')
            plcs, futures = extract_lightcones(padded_data, T, X,
                                            self.past_depth,
                                            self.future_depth,
                                            past_size,
                                            future_size,
                                            padding,
                                            self.c,
                                            base_anchor)
            # self.extractor.gather(field)
            # plcs, futures = (self.extractor.plcs, self.extractor.futures)
            new_map = collect_topological_morphs(plcs, futures)
            hierarchical_topological_merge(self.current_map, new_map, self.known_plcs)

            # Update morph size statistics
            morph_size = np.sum([len(morph) for morph in self.current_map.values()])
            self.morph_sizes[self.current_stage].append(morph_size)

        self.extractor.morph_map = {}


    def scan_stats(self):
        '''
        Prints the aggregate time spent gathering lightcone morphs from input fields
        and the total number of lightcone pairs scanned so far.
        '''
        print("{} total minutes of scan time; {} total lightcone pairs scanned".format(
        self.scan_time, self.total_pairs))

    def next_stage(self):
        '''
        Advances the Reconstructor to the next stage of the hierarchical
        reconstruction. This locks in the known past lightcones from the current
        stage which are to be ignored in the next stage. Make sure you have seen
        all lightcone pairs from the current stage before advancing, otherwise you
        will have to start over from the beginning.
        '''
        # Lock in known plcs
        self.known_plcs |= set(self.current_map.keys())
        # Advance curent stage variable
        self.current_stage += 1
        # Add empty map to list of morph maps for each stage
        self.morph_maps.append(defaultdict(set))
        # Set this to be the current map
        self.current_map = self.morph_maps[self.current_stage]
        # Add empty list to list of morph size lists
        self.morph_sizes.append([])

        print("Reconstructor had been in stage {}, and is now in stage {}".format(\
        str(self.current_stage - 1), str(self.current_stage)))

    def reset(self):
        '''
        If something has gone wrong and you need to start the hierarchical
        reconstruction again scratch. Can call this method rather than creating
        a new Reconstructor instance.
        '''
        self.morph_maps = [defaultdict(set)]
        self.morph_sizes = [[]]
        self.current_stage = 0
        self.known_plcs = set()
        self.current_map = self.morph_maps[self.current_stage]

        self.machine = None

    def plot_statistics(self, stage=None, size=10):
        '''
        Simple matplotlib plot of flc and V morph sizes as a funciton of number
        of spacetime fields scanned.
        '''
        if stage > self.current_stage:
            raise ValueError("Invalid stage; current stage is {}. Use current \
            stage or lower.".format(self.current_stage))
        elif stage is None:
            stage = self.current_stage
        plt.figure(figsize = (size, size))
        plt.plot(self.morph_sizes[stage], label = '{}'.format(self.future))
        plt.legend()
        plt.show()

    def reconstruct_states(self, stage=None):
        '''
        Uses the current main lists to reconstruct the states of a
        topological_machine object.
        '''
        # Raise ValueError if calling for stage which has not been reconstructed
        if stage > self.current_stage:
            raise ValueError("Invalid stage; current stage is {}. Use current \
            stage or lower.".format(self.current_stage))
        elif stage is None:
            stage = self.current_stage
        # Initialize a topological machine instance
        self.machine = TopologicalMachine(self.past_depth, self.future_depth,
                                            self.c)
        # Iteratively run construct states for each morph map up to the desired stage
        for morph_map in self.morph_maps[:stage+1]:
            self.machine.construct_states(morph_map)

    def reconstruct_transitions(self):
        '''
        Uses the current main lists to reconstruct the transitions of a
        topological_machine object.
        '''

    def reconstruct_machine(self, future):
        '''
        Uses the current main lists to reconstruct the states and transitions of a
        topological_machine object.
        '''

###################################################################################################

def load_reconstructor(past_depth, future_depth, c, saved_morph_sizes, saved_map):
    '''
    Takes data generated on the cluster to instantiate a TopologicalReconstructor object on a local machine
    '''
    # instantiate reconstructor
    recon = TopologicalReconstructor(past_depth, future_depth, c)
    # load merge morph sizes
    with open(saved_morph_sizes, "rb") as input_file:
         merge_sizes = pickle.load(input_file)
    recon.merge_morph_sizes = merge_sizes
    # load merge morph map
    with open(saved_map, "rb") as input_file:
        merge_map_compressed = pickle.load(input_file)
    merge_map = uncompress_dictionary(merge_map_compressed)
    recon.main_map = merge_map

    return recon

###################################################################################################

def enumerated_reconstruction(rule_number, past_depth, future_depth, transient=1, map_only=False):
    '''
    For ECAs only (if I can get a general 1D CA simulator working with numba I can use that here).

    Numba spacetime field function instead of ECA class

    Uses numba instead of numpy for lightcone extraction

    Still uses dictionary for morph map.

    Optional transient time option, where transients increase spatial extent of source fields to account for boundary conditions,
    so extra transient time has exponential computional cost.
    '''
    # initialze stuff
    alphabet_size = 2
    radius = 1

    time_steps = past_depth + future_depth + transient
    length = source_string_length(past_depth, future_depth, radius) + 2*radius*transient
    t, x = (past_depth + transient, int(length/2))
    morph_map = defaultdict(set)
    source_strings = product(np.arange(alphabet_size), repeat=length)
    lookup = lookup_table(rule_number, alphabet_size, radius)

    # iterate through source strings
    for source_string in source_strings:
        # create source spacetime field, get the lightcones, and update morph map
        string_array = np.array(source_string)
        source_field = numba_spacetime(string_array, lookup, time_steps)

        plc, flc = grab_lightcones(source_field, t, x, past_depth, future_depth, radius)

        morph_map[tuple(plc)].add(tuple(flc))

    if map_only:
        return morph_map
    else:
        # create the topological machine using the enumerated morph map
        machine = TopologicalMachine(past_depth, future_depth, radius)
        machine.construct_states(morph_map)
        return machine
