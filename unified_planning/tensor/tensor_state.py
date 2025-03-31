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

#from tensorflow.lookup.experimental import MutableHashTable
from tensorflow.lookup.experimental import DenseHashTable




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

        self._problem_metric=self.problem.quality_metrics[0]
        self.str_metric_expr=str(self._problem_metric.expression)
        self.tf_metric_expr=tf.constant(self.str_metric_expr, tf.string)

        if initialize==True:
            self._initialize_state()   


    def insert_zero(self, keys):

        pos=len(self._state)
        count=0
        for k in keys:
            k_str=str(k)
            if k_str in self._state:
                continue

            fluent=self._create_fluent(k, 0.0, False)
            self._state[k_str] = fluent
            if k_str not in self._keys_positions:
                self._keys_positions[k_str]=pos
                
            pos+=1
            count+=1

        #update lists
        #if count >0:
        #    self.compute_keys_positions()

    def set_value(self, pos, value):
        self.values_tensor[pos].assign(value)
    
    def compute_keys_positions(self):
        curr_state=self.get_tensors()
        self.keys_tensor = tf.constant(list(curr_state.keys()), dtype=tf.string)
        self.values_tensor =tf.Variable(list(curr_state.values()), dtype=tf.float32) 
        self.initial_values_tensor = tf.Variable(self.values_tensor)
        GlobalData._class_values_tensor=self.values_tensor
        pos=0
        self._keys_positions={}
        for k in curr_state.keys():
            self._keys_positions[str(k)]=pos
            pos+=1
        self.pos_metric_expr=self.get_key_position(self.str_metric_expr)
        self.pos_are_prec_sat=self.get_key_position(ARE_PREC_SATISF_STR)
        GlobalData.pos_metric_expr=tf.constant(self.pos_metric_expr)
        GlobalData.pos_are_prec_sat=tf.constant(self.pos_are_prec_sat)
        

    def get_value(self, key):
        pos=self.get_key_position(key)
        if pos==MISSING_VALUE:
            return TF_ZERO
        else:
            return self.values_tensor[pos]
        
    def get_key_position(self, key):
        return self._keys_positions.get(key, MISSING_VALUE)

    def get_key(self, pos):
        #if pos >=0 and pos< len(self.keys_tensor):
            return self.keys_tensor[int(pos)]
        #else:
        #    return MISSING_VALUE
        
    def get_initial_state_values(self):

        self.values_tensor.assign(self.initial_values_tensor)
        return self.values_tensor
    
    def get_state_values(self):
        return self.values_tensor

    def _initialize_state(self):
        """
        Internal method to initialize the state based on the problem's initial values.
        To be overridden in subclasses if needed.
        """
        for fluent, value in self.problem.initial_values.items():
            #print("Fluent:", fluent)
            #print("Initial value:", value)
            fluent_name=fluent.get_name()
            if fluent_name == "large_container" or fluent_name == "small_container":
                trainable=True
            else:  
                trainable=False
            if not (fluent.type.is_bool_type() and value.is_true() == False):
                self._state[fluent_name] = self._create_fluent(fluent, value, trainable) # XXXX check if 
            if trainable:
                self._trainable_fluents.append(fluent_name)
            else:
                self._non_trainable_fluents.append(fluent_name)
        # Check for metric_expression
        if self._state.get(self.str_metric_expr, None) is None:
            self._state[self.str_metric_expr] = self._create_fluent(self.str_metric_expr, TF_ZERO, False)
            self._non_trainable_fluents.append(self.str_metric_expr)
        
        #Add the are_preconditions_satisfied fluent
        self._state[ARE_PREC_SATISF_STR] = self._create_fluent(ARE_PREC_SATISF_STR, TF_ZERO, False)
        self._non_trainable_fluents.append(ARE_PREC_SATISF_STR)
        #update lists
        self.compute_keys_positions()
                
    
    def get_state(self):
        """
        Returns the current state.

        :return: Dictionary of fluent names and their corresponding values.
        """
        return self._state

 
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
        for key, value in state._state.items():
            if  value!=0 and not any(excluded in key for excluded in excluded_list) :
                if value is tf.Tensor:
                    print_val= value.numpy()
                else:
                    print_val=value
                
                print(key,": ", value, end=" -- ")

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
