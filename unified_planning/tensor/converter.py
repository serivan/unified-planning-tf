
from abc import ABC, abstractmethod

import os
import sympy
import sympy as sp
import tensorflow as tf
import unified_planning as up

from typing import Union

from unified_planning.tensor.constants import *
#from unified_planning.tensor import TensorAction

from unified_planning.model import OperatorKind
from unified_planning.model import EffectKind

class SympyToTensorConverter(ABC):
    """
    A utility class to handle the conversion of SymPy expressions to TensorFlow operations.
    """

    def __init__(self, state: up.model.State): #check use:: tensor_action:unified_planning.tensor.TensorAction
        self.state = state  # The state provides the values for the SymPy variables.


    def convert(self, sympy_expr):
        """
        Convert a SymPy expression to a TensorFlow tensor by recursively evaluating it.

        Args:
            sympy_expr (sympy.Expr): The SymPy expression to convert.

        Returns:
            TensorFlow Tensor: The resulting TensorFlow tensor.
        """

        if(DEBUG>6):
            print("..convert")
        result= self._tensor_convert(sympy_expr)
        if(DEBUG>6):
            print("Converter Result:", result)
        return result
    
    def set_state(self, state):
        ''' Set the state to be used for conversion
            Args: state (State): The state to use
        ''' 
        self.state = state

    def compute_effect_value(self, effect, are_prec_satisfied=0.0):
        ''' Compute the value of the effect
            Args:   effect (Effect): The effect to apply.
            Returns: The value of the effect
        '''
        
        if(DEBUG>6):
            print("..compute_effect_value")
        if effect.kind == EffectKind.ASSIGN:
            #return str(effect.value)
            if(are_prec_satisfied<=0.0):
                value_str= str(effect.fluent.get_name())
            else:
                #if effect.value.node_type==OperatorKind.INT_CONSTANT or effect.value.node_type==OperatorKind.REAL_CONSTANT or effect.value.node_type==OperatorKind.BOOL_CONSTANT:
                #    return effect.value.constant_value()
                value_str= str(effect.value) #+"+0.0"
            #value_str= str(effect.fluent.get_name()) +'* (1-'+str(are_prec_satisfied)+')'+'+'+ str(effect.value)+'*'+ str(are_prec_satisfied)
        elif effect.kind == EffectKind.INCREASE:
            value_str= str(effect.fluent.get_name())
            if(are_prec_satisfied>0.0):
                value_str= value_str +'+'+ str(effect.value)
        elif effect.kind == EffectKind.DECREASE:
            value_str= str(effect.fluent.get_name())
            if(are_prec_satisfied>0.0):
                value_str= value_str +'-'+ str(effect.value)+'*'+ str(are_prec_satisfied)
        else:
            raise ValueError("Unsupported effect kind")
        if(DEBUG>=0):
            tf.print("Effect value: ", value_str)
            os.sync()
        
        sympy_expr = sympy.sympify(value_str)
        
        result = self.convert(sympy_expr)      
        
        # Dynamically evaluate the operation string in the given scope
        # Add TensorFlow's namespace to the evaluation environment
        #local_scope = {**self.state, "tf": tf}
        #result= eval(value_str, {}, local_scope)
        return result

    
    def compute_condition_value(self, condition: Union[
            "up.model.fnode.FNode",
            "up.model.fluent.Fluent",
            "up.model.parameter.Parameter",
            bool]):
        value_str = ""
        if(DEBUG>6):
            print("..compute_condition_value")
        if condition.node_type == OperatorKind.LT:
            value_str = str(condition.args[1])+'-'+str(condition.args[0])
        elif condition.node_type == OperatorKind.LE:
            value_str = str(condition.args[1])+'-'+str(condition.args[0])+ '+' +str(TENSOR_EPSILON)
        elif condition.node_type == OperatorKind.FLUENT_EXP:
            value_str = str(condition.get_name())   
        elif condition.node_type == OperatorKind.NOT:
            cond_value=self.compute_condition_value(condition.args[0]) 
            return 1.0- cond_value
        elif condition.node_type == OperatorKind.AND:
            result=0.0
            for arg in condition.args:
                if arg.type.is_bool_type()==True:
                    value=self.compute_condition_value(arg) - TENSOR_EPSILON #to avoid 0.0
                else:
                    value=self.compute_condition_value(arg)
                val=self.negativeRelu( value )
                result += val
            if result >= 0.0:
                result=1.0 -result #result=0 when all conditions are satisfied, negative when one is not satisfied; a new result tensor value of 1.0 means the condition is satisfied
            
            return result
        elif condition.node_type == OperatorKind.OR:
            result=1.0
            for arg in condition.args:
                if arg.type.is_bool_type():
                    value=self.compute_condition_value(arg) - TENSOR_EPSILON #to avoid 0.0
                else:
                    value=self.compute_condition_value(arg)
                val= (-1.0) * self.negativeRelu( value ) # value=0 when the condition is satisfied, negative when it is not satisfied
                result *= val
            if result >= 0.0:
                result= (-1.0) * result #a negative tensor value means the condition is NOT satisfied
            else:
                result=1.0-result #results=0 and a tensor value of 1.0 means the condition is satisfied
            return result
        else:
            raise ValueError("Operator not supported")
        
        sympy_expr = sympy.sympify(value_str)
        result = self.convert(sympy_expr)
        
        # Dynamically evaluate the operation string in the given scope
        # Add TensorFlow's namespace to the evaluation environment
        #local_scope = {**self.state, "tf": tf}
        #result= eval(value_str, {}, local_scope)
        return result


    def _tensor_convert(self, node):
        """
        Recursively convert a SymPy node into a TensorFlow/Torch operation.

        Args:
            node (sympy.Basic): A SymPy expression node to convert.

        Returns:
            TensorFlow/Torch Tensor: The resulting tensor.
        """
        node_name=str(node)
        if(DEBUG>6):
            print("..tensor_convert")
        if isinstance(node, sympy.Symbol):
            # Look up the symbol in the state and return its value
            if node_name in self.state:
                value=self.state[node_name]
            else:
                value=0.0
            if DEBUG>5:
                print(node_name,"= ", value)
            return TF_ZERO
        elif isinstance(node, sympy.Number):
            if self.is_tensor(node):
                return node
            else:
                # Return the number as a TensorFlow/Torch constant
                return self.get_constant(node) 
        elif isinstance(node, sympy.Basic):
            # Handle basic SymPy operations (Add, Mul, Pow, etc.)
            if isinstance(node, sympy.Add):
                args=[self._tensor_convert(arg) for arg in node.args]
                
                result=tf.add_n([arg for arg in args])
                #result= lambda: (tf.add_n([arg() if callable(arg) else arg for arg in args]), args)
                return result
            elif isinstance(node, sympy.Mul):
                args = [self._tensor_convert(arg) for arg in node.args]
                
                result=1.0
                for arg in args:
                    result *= arg
                return result
            elif isinstance(node, sympy.Pow):
                func = self.sympy_to_tensor_map.get(type(node))
                base, exp = [self._tensor_convert(arg) for arg in node.args]
                return func(base, exp)
            elif  node_name in self.state:
                value = self.state[node_name]
                return value
            else:
                func = self.sympy_to_tensor_map.get(type(node))
                if func:
                    return func(*[self._tensor_convert(arg) for arg in node.args])
                else:

                    raise ValueError(f"Unsupported operation: {type(node)}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    @abstractmethod
    def negativeRelu(self, x):
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def get_constant(self, node):
        raise NotImplementedError("Subclasses should implement this method")
    @abstractmethod
    def is_tensor(self, node):
        raise NotImplementedError("Subclasses should implement this method")

class SympyToTfConverter(SympyToTensorConverter):
    def __init__(self, state: up.model.State):
        super().__init__(state)
        self.sympy_to_tensor_map = {
            sp.sin: tf.sin,
            sp.cos: tf.cos,
            sp.exp: tf.exp,
            sp.log: tf.math.log,
            sp.Max: tf.maximum,  # ReLU equivalent for Max
            sp.Pow: tf.pow,      # Add the pow function mapping
        }

    def negativeRelu(self, x):
        """
        Applies the negative ReLU activation function to the input tensor.

        The negative ReLU activation function is defined as the ReLU activation
        applied to the negation of the input tensor. This means that all positive
        values in the input tensor will be set to zero, and all negative values
        will be converted to their positive counterparts.

        Parameters:
        x (tf.Tensor): The input tensor to which the negative ReLU activation
                       function will be applied.

        Returns:
        tf.Tensor: A tensor with the negative ReLU activation function applied.
        """
        return (-1.0)*tf.keras.activations.relu(-x)
    
    def get_constant(self, node): 
        return tf.constant(float(node), dtype=tf.float32)
    
    def is_tensor(self, node):
        return node is tf.Tensor
    