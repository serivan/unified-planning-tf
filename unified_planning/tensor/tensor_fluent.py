from abc import ABC, abstractmethod
import tensorflow as tf

from unified_planning.tensor.constants import *
from fractions import Fraction

class TensorFluent(ABC):
    _global_fluent_id = 0  # Class attribute to keep track of the global ID

    def __init__(self, up_fluent, value, trainable=False, _activation_function=None):
        """
        Initialize the TensorFluent with a name.

        Args:
            name (str): The name of the fluent.
        """
        self._up_fluent =None
        if isinstance(up_fluent, str):  
            self.name = up_fluent
        else:
            self._up_fluent = up_fluent
            self.name = up_fluent.get_name()
        self._fluent_value = None
        self._value_node= value
        self._trainable = trainable
        self._activation_function = self._define_activation_function(_activation_function)
        
        self._fluent_id = TensorFluent._global_fluent_id   
        TensorFluent._global_fluent_id += 1

    #@tf.function
    def get_name(self):
        """
        Get the name of the fluent.

        Returns:
            str: The name of the fluent.
        """
        return self.name

    @abstractmethod
    def _define_activation_function(self, _activation_function):
        """
        Get the value of the fluent.

        Returns:
            tf.Variable: The value of the fluent.
        """
        pass
    
    def get_fluent_id(self):
        """
        Get the ID of the fluent.

        Returns:
            int: The ID of the fluent.
        """
        return self._fluent_id

    @abstractmethod
    def get_activation_function(self):
        """
        Get the value of the fluent.

        Returns:
            tf.Variable: The value of the fluent.
        """
        pass
    
    @abstractmethod
    def get_tensor(self):
        """
        Get the value of the fluent.

        Returns:
            tensor: The value of the fluent.
        """
        pass
    
    @abstractmethod
    def get_value(self):
        """
        Get the value of the fluent.

        Returns:
            tf.Variable: The value of the fluent.
        """
        pass

    @abstractmethod
    def set_value(self, new_value):
        """
        Set a new value for the fluent.

        Args:
            new_value (tf.Variable): The new value to set.
        """
        pass

    def __repr__(self):
        """
        Return a string representation of the TensorFluent for debugging purposes.

        Returns:
            str: The string representation of the TensorFluent.
        """
        return f"TensorFluent(name={self.name})"
    
    def __eq__(self, other):
        """
        Compare two TensorFluent objects for equality.

        Args:
            other (TensorFluent): The other TensorFluent object to compare with.

        Returns:
            bool: True if the two TensorFluent objects are equal, False otherwise.
        """
        if not isinstance(other, TensorFluent):
            return False
        return self._fluent == other._fluent and tf.reduce_all(tf.equal(self.get_value(), other.get_value()))  and self._activation_function == other._activation_function


    def __hash__(self):
        """
        Generate a hash value for the TensorFluent object.

        Returns:
            int: The hash value of the TensorFluent object.
        """
        return hash((self.name, tuple(self.get_value().numpy().flatten()), self._activation_function))



class TfFluent(TensorFluent):
    def __init__(self, up_fluent, value_node, trainable=False):
        """
        Initialize the TfFluent with a name and a TensorFlow variable.

        Args:
            fluent: The UP fluent.
            value (tf.Variable): The value of the fluent.
        """
        super().__init__(up_fluent, value_node, trainable)
#        if value_node.is_bool_constant():
#            self._fluent_value = tf.Variable(value_node.constant_value(), dtype=tf.bool, trainable= _trainable)
      
        if isinstance(value_node, tf.Tensor):
            self._fluent_value = value_node
        else:
            if isinstance(value_node,int) or isinstance(value_node,float):
                value=value_node
            else:
                value=value_node.constant_value()

            if isinstance(value, Fraction):
                value=float(value)
                
            self._fluent_value = tf.Variable(value, dtype=tf.float32, trainable=trainable)


    def _define_activation_function(self, _activation_function):
        """
        Get the value of the fluent.

        Returns:
            tf.Variable: The value of the fluent.
        """
        if _activation_function is None:
            return tf.identity
        elif self._value_node.is_bool_constant() :
            return TF_ACTV_FN_BOOL
        else:
            return TF_ACTV_FN_REAL
            


    def get_activation_function(self):
        """
        Get the value of the fluent.

        Returns:
            tf.Variable: The value of the fluent.
        """
        return self._activation_function    

    def get_tensor(self):
        """
        Get the value of the fluent.

        Returns:
            tf.Variable: The value of the fluent.
        """
        return self._fluent_value


    def get_value(self):
        """self.value = tf.Variable(value, trainable=trainable)
        Get the value of the fluent.

        Returns:
            tf.Variable: The value of the fluent.
        """
        return self._activation_function(self._fluent_value)

    def set_value(self, new_value):
        """
        Set a new value for the fluent.

        Args:
            new_value (tf.Variable): The new value to set.
        """
        if isinstance(self._fluent_value, tf.Variable):
            self._fluent_value.assign(new_value)
        elif isinstance(new_value, tf.Tensor):
            self._fluent_value=new_value
        else:
            self._fluent_value= tf.constant(new_value, dtype=tf.float32)

    def __repr__(self):
        """
        Return a string representation of the TfFluent for debugging purposes.

        Returns:
            str: The string representation of the TfFluent.
        """
        #return f"TfFluent(name={self.name}, fluent_value={self._fluent_value.numpy()}, value={self.get_value()})"
        return f"TfFluent(name={self.name}, fluent_value={self._fluent_value}, value={self.get_value()})"
