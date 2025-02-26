from abc import ABC, abstractmethod
from typing import Union, Optional, List, Dict

import tensorflow as tf
#import torch

import unified_planning as up
from unified_planning.model import OperatorKind
from unified_planning.tensor.tensor_fluent import TfFluent
from unified_planning.tensor.constants import *
from unified_planning.model import OperatorKind
from unified_planning.model import EffectKind

from tensorflow.lookup.experimental import MutableHashTable




class TensorState(ABC):
    def __init__(self, problem, initialize=True):
        """
        Initializes the TensorState object by setting up the initial state.

        :param problem: The problem instance containing initial values.
        """
        self.problem = problem
        
        self._state = {}  # Dictionary to store the initial state
        self._trainable_fluents= [] # List to store the trainable fluents
        self._non_trainable_fluents= [] # List to store the non-trainable fluents
        self.action=None
        self._restore_keys=None
        # Populate the initial state (delegated to a subclass, if necessary)
        if initialize==True:
            self._initialize_state()   

        
        # Create a MutableHashTable
        default_value = MISSING_VALUE  # Default value if a key is not found
        self._hash_state = MutableHashTable(
            key_dtype=tf.string, 
            value_dtype=tf.float32, 
            default_value=default_value
        )
        self._initial_hash_state = MutableHashTable(
            key_dtype=tf.string, 
            value_dtype=tf.float32, 
            default_value=default_value
        )
        curr_state=self.get_tensors()
        # Convert dictionary keys and values to TensorFlow tensors
        keys_tensor = tf.constant(list(curr_state.keys()), dtype=tf.string)
        values_tensor =tf.Variable(list(curr_state.values()), dtype=tf.float32) 

        # Insert data into the hash table
        self._hash_state.insert(keys_tensor, values_tensor)
        # Insert data into the curr_hash table
        self._initial_hash_state.insert(keys_tensor, values_tensor)

 
    def _initialize_state(self):
        """
        Internal method to initialize the state based on the problem's initial values.
        To be overridden in subclasses if needed.
        """
        for fluent, value in self.problem.initial_values.items():
            #print("Fluent:", fluent)
            #print("Initial value:", value)
            if fluent.get_name() == "large_container" or fluent.get_name() == "small_container":
                trainable=True
            else:  
                trainable=False
            if not (fluent.type.is_bool_type() and value.is_true() == False):
                self._state[fluent.get_name()] = self._create_fluent(fluent, value, trainable) # XXXX check if 
            if trainable:
                self._trainable_fluents.append(fluent.get_name())
            else:
                self._non_trainable_fluents.append(fluent.get_name())
        
        #Add the are_preconditions_satisfied fluent
        #self._non_trainable_fluents.append(ARE_PREC_SATISF_STR)
        
    def restore_hash_state(self, restore_keys=None): 
        if restore_keys is None:
            if self._restore_keys is None:
                restore_keys = self._initial_hash_state.export()[0]
            else:
                restore_keys = self._restore_keys  
        else:
            self._restore_keys = restore_keys
        #            restore_keys = self._initial_hash_state.export()[0]
        #            restore_values = self._initial_hash_state.export()[1]
         
        resore_values=self._initial_hash_state.lookup(restore_keys)
        self._hash_state.insert(restore_keys, resore_values)
    
    def get_hash_state(self):
        """
        Returns the current state.

        :return: Dictionary of fluent names and their corresponding values.
        """
        return self._hash_state
        
    
    def get_state(self):
        """
        Returns the current state.

        :return: Dictionary of fluent names and their corresponding values.
        """
        return self._state

 
    
    def get_hash_state(self):
        """
        Returns the current state.

        :return: Dictionary of fluent names and their corresponding values.
        """
        return self._hash_state
    
    def get_initial_hash_state(self):
        """
        Returns the current state.

        :return: Dictionary of fluent names and their corresponding values.
        """
        return self._initial_hash_state

    def __ref__(self):
        """
        Prints the current state for debugging purposes.
        """
        for fluent, variable in self._state.items():
            print(f"{fluent}: {variable}")


    def copy(self):
        """
        Create a shallow copy of the current state.

        Returns:
            TensorState: A new instance of TensorState with the same fluents and values.
        """
        new_state = self.__class__(self.problem, False)
        # Deep copy new_state.initial_state =  {fluent: value.copy() for fluent, value in self.initial_state.items()}
        new_state._state = self._state.copy() 
        return new_state


    def __getitem__(self, key):
        """
        Get the value of the fluent by key.

        Args:
            key (str): The name of the fluent.

        Returns:
            TfFluent: The fluent associated with the given key.
        """
        return self._state[key]
    
    @abstractmethod
    def set_attr(self, key, value):
        """
        Set the value of the fluent by key.

        Args:
            key (str): The name of the fluent.
            value (TfFluent): The new value to set.
        """
        pass
        
    def __setitem__(self, key, value):
        """
        Set the value of the fluent by key.

        Args:
            key (str): The name of the fluent.
            value (TfFluent): The new value to set.
        """
        self._state[key] = value

    def __eq__(self, other):
        """
        Compare two TensorState objects for equality.

        Args:
            other (TensorState): The other TensorState object to compare with.

        Returns:
            bool: True if the two TensorState objects are equal, False otherwise.
        """
        if not isinstance(other, TensorState):
            return False
        return self._state == other._state

    def __hash__(self):
        """
        Generate a hash value for the TensorState object.

        Returns:
            int: The hash value of the TensorState object.
        """
        return hash(frozenset(self._state.items()))

    @abstractmethod
    def _create_fluent(self, fluent, value, trainable):
        """
        Create a new fluent object with the given name and value.

        Args:
            fluent (Fluent): The fluent object to create.
            value (tf.Variable): The value of the fluent.

        Returns:
            TensorFluent: A new instance of the TensorFluent object.
        """
        pass


    def get_tensors(self):
        """
        Convert the TensorState object to a dictionary.

        Returns:
            dict: A dictionary representation of the TensorState object.
        """
        return {fluent: value.get_tensor() for fluent, value in self._state.items()}

    
    def __repr__(self):
        """
        Prints the current state for debugging purposes.
        """
        result="TensorState:("
        for fluent, variable in self._state.items():
            result+=f"{fluent}: {variable},"
        result+=")"
        return result

   
    @staticmethod
    def check_dict_state(state, string):
        """
        Check the state values.

        :param state: The state to check.
        :param string: The string to print.
        """
        it=0
        for fluent, value in state.items():
            it+=1
            if fluent == string:
                tf.print(fluent, value)

    @staticmethod
    def copy_dict_state_in_new_state(state, new_state):
        """
        Copy the state in the new_state dictionary.

        :param state: The state to copy.
        :param new_state: The new state dictionary already existing with the same structure.
        """
        it=0
        string='__builtins__'
        for fluent, value in state.items():
            it+=1
            if fluent == string:
                tf.print(fluent, value)
            new_state[fluent].assign(value)
            
        #self.check_state(new_state, '__builtins__')
    @staticmethod
    def shallow_copy_dict_state(state):
        """
        Create a shallow copy of the state.
    
        :param state: The state to copy.
        :return: A new state dictionary with the same structure and values.
        """
        new_state = {}
        for key, value in state.items():
            new_state[key] = value
        return new_state

    @staticmethod
    def create_copy_dict_state(state):
        """
        Create a new state dictionary with the same structure as the input state.

        :param state: The state to copy.
        :return: The new state dictionary with the same structure.
        """
        new_state = {}
        for fluent, value in state.items():
            new_state[fluent] = tf.Variable(value, trainable=value.trainable, dtype=tf.float32)

        self.check_state(new_state, '__builtins__')
        return new_state


    @staticmethod
    def print_filtered_dict_state(state, excluded_list=None):
        for key, value in state.items():
            if  value!=0 and not any(excluded in key for excluded in excluded_list) :
                if value is tf.Tensor:
                    print_val= value.numpy()
                else:
                    print_val=value
                
                print(key,": ", value, end=" -- ")

    @staticmethod
    def print_filtered_hash_state(state, excluded_list=None):

        keys, values = state.export()  # Get all keys and values
        
        for key, value in zip(keys, values):
            if  value!=0 and not any(excluded in str(key) for excluded in excluded_list) :
                if value is tf.Tensor:
                    print_val= value.numpy()
                else:
                    print_val=value
                
                print(str(key.numpy()),": ", float(print_val), end=" -- ")


    def print_mutable_hash_table(hash_table):
        """
        Print the contents of a MutableHashTable.

        Args:
            hash_table: The MutableHashTable to print.
        """
        keys = hash_table.export()[0]
        values = hash_table.export()[1]

        keys_values = tf.stack([keys, values], axis=1)
        tf.print("MutableHashTable contents:\n", keys_values)

    @staticmethod
    def shallow_copy_hash_state(hash_state):
        """Creates a shallow copy of a tf.lookup.MutableHashTable.
    
            Args:
                hash_table (tf.lookup.MutableHashTable): The hash table to copy.
        
            Returns:
                tf.lookup.MutableHashTable: A new hash table with the same key-value pairs.
        """
        # Get all keys from the existing hash table
        keys_tensor = hash_state.export()[0]
    
        # Lookup values for the extracted keys
        values_tensor = hash_state.lookup(keys_tensor)
    
        # Create a new hash table with the same default value
        default_value = MISSING_VALUE  # Default value if a key is not found
        new_hash_state =  MutableHashTable(
            key_dtype=hash_state.key_dtype, 
            value_dtype=hash_state.value_dtype, 
            default_value=default_value
        )
    
       # Insert the copied keys and values into the new table
        new_hash_state.insert(keys_tensor, values_tensor)
    
        return new_hash_state
    
    @staticmethod
    def print_hash_state(hash_table: MutableHashTable) -> None:
        """Retrieve and print all key-value pairs."""

        keys, values = self._hash_table.export()  # Get all keys and values
        
        for key, value in zip(keys.numpy(), values.numpy()):
            print(f"{key.decode()}: {value} -- ", end="")

class TfState(TensorState): #, tf.experimental.ExtensionType):
    def __init__(self, problem, intialize=True):
        """
        Initializes the TfState object by setting up the initial state.

        :param problem: The problem instance containing initial values.
        """
        super().__init__(problem, intialize)
        

    def _create_fluent(self, up_fluent, value, trainable=False):
        """
        Create a new fluent object with the given name and value.

        Args:
            fluent (Fluent): The fluent object to create.
            value (tf.Variable): The value of the fluent.

        Returns:
            TfFluent: A new instance of the TfFluent object.
        """
        return TfFluent(up_fluent, value, trainable)
    
    def set_attr(self, key, value):
        """
        Set the value of the fluent by key.

        Args:
            key (str): The name of the fluent.
            value (TfFluent): The new value to set.
        """

        if self._state[key] is None:
            raise ValueError("Fluent not found")
            #self._state[key]= create_fluent(fluent, value)
        elif isinstance(value, TfFluent):
            self._state[key]._fluent_value = value
        else:
            if isinstance(self._state[key]._fluent_value, tf.Variable):
                self._state[key]._fluent_value.assign(value)
            else:
                self._state[key].set_value(value)
