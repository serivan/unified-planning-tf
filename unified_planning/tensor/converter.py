
from abc import ABC, abstractmethod

import os
import sympy
import sympy as sp
import tensorflow as tf
import unified_planning as up

from typing import Union

from unified_planning.tensor.constants import *

from unified_planning.model import OperatorKind
from unified_planning.model import EffectKind

from unified_planning.tensor.constants import *

sympy_to_tensor_map = {
            sp.sin: tf.sin,
            sp.cos: tf.cos,
            sp.exp: tf.exp,
            sp.log: tf.math.log,
            sp.Max: tf.maximum,  # ReLU equivalent for Max
            sp.Min: tf.minimum,  # ReLU equivalent for Max
            sp.Pow: tf.pow,      # Add the pow function mapping
        }


def tensor_convert(node, are_prec_satisfied, state):
        """
        Recursively convert a SymPy node into a TensorFlow operation.
        
        Args:
            node (sympy.Basic): A SymPy expression node to convert.
        
        Returns:
            TensorFlow Tensor: The resulting tensor.
        """
        node_name = str(node)
        if DEBUG > 4:
            print("..tensor_convert:", node_name)
        
        if isinstance(node, tf.Tensor):
            value = node
        elif isinstance(node, sympy.logic.boolalg.BooleanTrue):
            value = 1.0 #tf.constant(1.0)
        elif isinstance(node, sympy.logic.boolalg.BooleanFalse):
            value = -1.0 #tf.constant(-1.0)
        elif isinstance(node, sympy.Symbol):
            if node_name == ARE_PREC_SATISF_STR:
                value = are_prec_satisfied
            else:
                value = state.lookup( tf.constant(node_name, tf.string)) 
        elif isinstance(node, sympy.Number):
            if node is tf.Tensor:
                value = node
            else:
                value = float(node)
        elif isinstance(node, sympy.Basic):
            if isinstance(node, sympy.Add):
                args = [tensor_convert(arg, are_prec_satisfied, state) for arg in node.args]
                value = tf.add_n(args)
            elif isinstance(node, sympy.Mul):
                args = [tensor_convert(arg, are_prec_satisfied, state) for arg in node.args]
                value = tf.math.reduce_prod(args)
            elif isinstance(node, sympy.Pow):
                func = sympy_to_tensor_map.get(type(node))
                base, exp = [tensor_convert(arg, are_prec_satisfied, state) for arg in node.args]
                value = func(base, exp)
            elif  tf.not_equal(self.state.lookup(tf.constant(node_name, tf.string)), MISSING_VAL): 
                if DEBUG > 5:
                    tf.print("Node:", node_name, "in state:", state[node_name])
                value = self.state.lookup(tf.constant(node_name, tf.string))
            else:
                func = sympy_to_tensor_map.get(type(node))
                if func:
                    value = func(*[tensor_convert(arg, are_prec_satisfied, state) for arg in node.args])
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
        
        if DEBUG > 5:
            tf.print(node_name, ":=", value)
        
        return value



class SympyToTensorConverter(ABC):
    """
    A utility class to handle the conversion of SymPy expressions to TensorFlow operations.
    """

    def __init__(self, state: up.model.State): #check use:: tensor_action:unified_planning.tensor.TensorAction
        self.state = state  # The state provides the values for the SymPy variables.

    

    def convert(self, sympy_expr,predicates_indexes, are_prec_satisfied=0.0):
        """
        Convert a SymPy expression to a TensorFlow tensor by recursively evaluating it.

        Args:
            sympy_expr (sympy.Expr): The SymPy expression to convert.

        Returns:
            TensorFlow Tensor: The resulting TensorFlow tensor.
        """

        if(DEBUG>4):
            print("..convert")
        result= self._tensor_convert(sympy_expr,predicates_indexes, are_prec_satisfied)
        #result= tensor_convert(sympy_expr, are_prec_satisfied, self.state)
        if(DEBUG>4):
            print("Converter Result:", result)
        return result
    
    def set_state(self, state):
        ''' Set the state to be used for conversion
            Args: state (State): The state to use
        ''' 
        self.state = state

    def when_sympy_expr_inserted(self, effect):
        return effect.sympy_expr

    def when_sympy_expr_not_inserted(self, effect, effects_set, are_prec_satisfied):
        # Handle different effect kinds
        if effect.kind == EffectKind.ASSIGN:
            value_str = str(effect.fluent.get_name()) if are_prec_satisfied < 0.0 else str(effect.value)

        elif effect.kind == EffectKind.INCREASE:
            value_str = str(effect.fluent.get_name()) if are_prec_satisfied < 0.0 else str(effect.fluent.get_name()) + '+' + str(effect.value)

        elif effect.kind == EffectKind.DECREASE:
            value_str = str(effect.fluent.get_name()) if are_prec_satisfied < 0.0 else str(effect.fluent.get_name()) + '-' + str(effect.value)
        elif  effect.fluent.node_type == OperatorKind.FLUENT_EXP and effect.value.is_bool_constant():
            value_str = str(effect.fluent.get_name()) if are_prec_satisfied < 0.0 else str(effect.value.bool_constant_value())
        else:
            raise ValueError("Unsupported effect kind")

        lifted_str=GlobalData.get_lifted_string(value_str,effects_set)
        # Convert to sympy expression and insert it
        sympy_expr = sympy.sympify(lifted_str)
        return sympy_expr

    def define_effect_value(self, effect, are_prec_satisfied=-1.0):
        ''' Compute the value of the effect
            Args:   effect (Effect): The effect to apply.
            Returns: The value of the effect
        '''

        value_str=""
        if(DEBUG>6):
            print("..compute_effect_value: ", effect.fluent.get_name())

        sympy_expr_sat = self.when_sympy_expr_not_inserted(effect, 1.0)
        
        sympy_expr_unsat = self.when_sympy_expr_not_inserted(effect, -1.0)
        
        effect.insert_sympy_expression(sympy_expr_sat, sympy_expr_unsat)

        if ( are_prec_satisfied ==1.0 ):
            sympy_expr = sympy_expr_sat
        else:   
            sympy_expr = sympy_expr_unsat


        result = self.convert(sympy_expr, are_prec_satisfied)      
        if(DEBUG>3):
            print("Effect value: ", value_str)
            print("Result: ", result)
            os.sync()

        # Dynamically evaluate the operation string in the given scope
        # Add TensorFlow's namespace to the evaluation environment
        #local_scope = {**self.state, "tf": tf}
        #result= eval(value_str, {}, local_scope)
        return result


    def compute_effect_value(self, effect, are_prec_satisfied=-1.0):
        ''' Compute the value of the effect
            Args:   effect (Effect): The effect to apply.
            Returns: The value of the effect
        '''

        if(DEBUG>6):
            print("..compute_effect_value: ", effect.fluent.get_name())

        if (are_prec_satisfied == 1.0):
            sympy_expr = effect.sympy_expr_sat
        else:
            sympy_expr = effect.sympy_expr_unsat
        
        result= self.convert(sympy_expr, are_prec_satisfied)

        if(DEBUG>3):
            print("Result: ", result)
            os.sync()
        return result

    def define_condition_str(self, condition: Union[
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
            return value_str #avoid simplify
        
        elif condition.node_type == OperatorKind.NOT:
            cond_value=self.define_condition_str(condition.args[0]) 
            value_str= "1.0 - " + cond_value
        elif condition.node_type == OperatorKind.AND:
            for arg in condition.args:
                if arg.type.is_bool_type()==True:
                    value=self.define_condition_str(arg) +" - " + str(TENSOR_EPSILON) #to avoid 0.0
                else:
                    value=self.define_condition_str(arg)
                
                val= "Min("+value+",0.0)" # value=0 when the condition is satisfied, negative when it is not satisfied
                if value_str == "":
                    value_str = "("+val+")"
                else:
                    value_str += "+ (" + val + ")"

        elif condition.node_type == OperatorKind.OR:
            for arg in condition.args:
                if arg.type.is_bool_type():
                    value=self.define_condition_str(arg) +" - " + str(TENSOR_EPSILON) #to avoid 0.0
                else:
                    value=self.define_condition_str(arg)
                val= "Max( (-1.0)* "+value+",0.0)" # value=0 when the condition is satisfied, negative when it is not satisfied; multiply for -1.0 in order to have all positive values when preconditionas are not satisied and at the end we multiply again for -1.0
                if value_str == "":
                    value_str = "("+val+")"
                else:
                    value_str += "* (" + val + ")"
            
            value_str+= " * (-1.0)" #a negative tensor value means the condition is NOT satisfied; previusly we have multiplied unsat preconditions for -1.0
            
        else:
            raise ValueError("Operator not supported")
        
        return value_str
        #sympy_expr = sympy.sympify(value_str)
        #return sympy_expr
  
    def get_condition_str(self, condition: Union[
            "up.model.fnode.FNode",
            "up.model.fluent.Fluent",
            "up.model.parameter.Parameter",
            bool]):
        
        value_str = "Min (0.0,"+ self.define_condition_str(condition)+")"  # value=0 when the condition is satisfied, negative when it is not satisfied
        return value_str 
        
    def define_condition_expr(self, condition_str: str):     
        sympy_expr = sympy.sympify(condition_str)
        return sympy_expr
      

    def compute_condition_value(self, condition: int):
        condition_str = self.define_condition_str(condition)
        sympy_expr =  self.define_condition_expr(condition_str)
        result = self.convert(sympy_expr)
        
        return result
    

    def compute_condition_value(self, condition: Union[
            "up.model.fnode.FNode",
            "up.model.fluent.Fluent",
            "up.model.parameter.Parameter",
            bool]):
        condition_str = self.define_condition_str(condition)
        sympy_expr =  self.define_condition_expr(condition_str)
        result = self.convert(sympy_expr)
        
        return result

    #@tf.function
    def _tensor_convert_old(self, node, predicates_indexes, are_prec_satisfied=tf.constant(1.0)):
        """
        Convert a SymPy node into a TensorFlow operation without recursion.

        Args:
            node (sympy.Basic): A SymPy expression node to convert.

        Returns:
            TensorFlow Tensor: The resulting tensor.
        """
        stack = [(node, None)]  # Stack to simulate recursion, (node, parent_index)
        results = {}  # Cache for computed values

        while stack:
            current, parent = stack.pop()

            if isinstance(current, tf.Tensor):
                results[id(current)] = current
            elif isinstance(current, sympy.logic.boolalg.BooleanTrue):
                results[id(current)] = tf.constant(1.0)
            elif isinstance(current, sympy.logic.boolalg.BooleanFalse):
                results[id(current)] = tf.constant(-1.0)
            elif isinstance(current, sympy.Symbol):
                node_name = str(current)
                lookup_key = tf.constant(node_name, dtype=tf.string)

                if node_name == ARE_PREC_SATISF_STR:
                    results[id(current)] = are_prec_satisfied
                elif node_name.startswith(LIFTED_STR):
                    results[id(current)] = self.extract_from_lifted(node_name, predicates_indexes)
                elif tf.not_equal(self.state.lookup(lookup_key), MISSING_VAL):
                    results[id(current)] = self.state.lookup(lookup_key)
                else:
                    results[id(current)] = tf.constant(-1.0)
            elif isinstance(current, sympy.Number):
                results[id(current)] = current if self.is_tensor(current) else self.get_constant(current)
            elif isinstance(current, sympy.Basic):
                if id(current) in results:
                    continue  # Skip if already processed

                args = list(current.args)
                missing_args = [arg for arg in args if id(arg) not in results]

                if missing_args:
                    stack.append((current, None))  # Re-process after args are computed
                    stack.extend((arg, current) for arg in missing_args)
                    continue

                # Compute the result once all arguments are available
                if isinstance(current, sympy.Add):
                    results[id(current)] = tf.add_n([results[id(arg)] for arg in args])
                elif isinstance(current, sympy.Mul):
                    results[id(current)] = tf.math.reduce_prod([results[id(arg)] for arg in args])
                elif isinstance(current, sympy.Pow):
                    base, exp = results[id(args[0])], results[id(args[1])]
                    results[id(current)] = self.sympy_to_tensor_map.get(type(current))(base, exp)
                elif tf.not_equal(self.state.lookup(tf.constant(str(current), tf.string)), MISSING_VAL):
                    results[id(current)] = self.state.lookup(tf.constant(str(current), tf.string))
                elif tf.cond(tf.equal(tf.strings.substr(tf.constant(str(current), tf.string), 0, len(LIFTED_STR)), LIFTED_STR), 
                             lambda: True, lambda: False):
                    results[id(current)] = self.extract_from_lifted(str(current), predicates_indexes)
                else:
                    func = self.sympy_to_tensor_map.get(type(current))
                    if func:
                        results[id(current)] = func(*[results[id(arg)] for arg in args])
                    else:
                        raise ValueError(f"Unsupported operation: {type(current)}")
            else:
                raise ValueError(f"Unsupported node type: {type(current)}")

        return results[id(node)]

    #@tf.function
    def _tensor_convert(self, node, predicates_indexes, are_prec_satisfied=tf.constant(1.0)):
        """
        Recursively convert a SymPy node into a TensorFlow operation.
        
        Args:
            node (sympy.Basic): A SymPy expression node to convert.
        
        Returns:
            TensorFlow Tensor: The resulting tensor.
        """
    
        if isinstance(node, tf.Tensor):
            value = node
        elif isinstance(node, sympy.logic.boolalg.BooleanTrue):
            value = tf.constant(1.0)
        elif isinstance(node, sympy.logic.boolalg.BooleanFalse):
            value = tf.constant(-1.0)
        elif isinstance(node, sympy.Symbol):
            
            node_name = node.name
            lookup_key = tf.constant(node_name, dtype=tf.string)
            if DEBUG > 4:
                print("..tensor_convert:", node_name)
            if node_name == ARE_PREC_SATISF_STR:
                value = are_prec_satisfied
            elif node_name.startswith(LIFTED_STR):
                value=self.extract_from_lifted(lookup_key, predicates_indexes)
            elif tf.not_equal(self.state.lookup(lookup_key), MISSING_VAL): 
                if DEBUG > 5:
                    print("Node:", node_name, "in state:", self.state[node_name])
                value = self.state.lookup(lookup_key)
            else:
                value = tf.constant(-1.0)
        elif isinstance(node, sympy.Number):
            value = node if self.is_tensor(node) else self.get_constant(node)
        elif isinstance(node, sympy.Basic):
            if isinstance(node, sympy.Add):
                args = [self._tensor_convert(arg, predicates_indexes,  are_prec_satisfied) for arg in node.args]
                value = tf.add_n(args)
            elif isinstance(node, sympy.Mul):
                args = [self._tensor_convert(arg, predicates_indexes, are_prec_satisfied) for arg in node.args]
                value = tf.math.reduce_prod(args)
            elif isinstance(node, sympy.Pow):
                func = self.sympy_to_tensor_map.get(type(node))
                base, exp = [self._tensor_convert(arg, predicates_indexes, are_prec_satisfied) for arg in node.args]
                value = func(base, exp)      
            #elif  tf.not_equal(self.state.lookup(lookup_key), MISSING_VAL):
            #    if DEBUG > 5:
            #        tf.print("Node:", node_name, "in state:", self.state[node_name])
            #    value = self.state.lookup(lookup_key)
            #elif   tf.equal(tf.strings.substr(lookup_key, 0, len(LIFTED_STR)), LIFTED_STR): #tf.cond( tf.equal(tf.strings.substr(lookup_key, 0, len(LIFTED_STR)), LIFTED_STR), lambda: True, lambda: False): 
            #    value=self.extract_from_lifted(lookup_key, predicates_indexes)    
            else:
                func = self.sympy_to_tensor_map.get(type(node))
                if func:
                    value = func(*[self._tensor_convert(arg,predicates_indexes,  are_prec_satisfied) for arg in node.args])
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
        
        if DEBUG > 5:
            print(node_name, ":=", value)
        
        return value

    
    def extract_from_lifted(self, node_name: tf.Tensor, predicates_indexes: tf.Tensor):
        pos_str = tf.strings.regex_replace(node_name, LIFTED_STR, "")
        pos = tf.strings.to_number(pos_str, out_type=tf.int32)  # Convert to integer
        if DEBUG > 5:
            tf.print("Pos: ", pos, "Predicates indexes: ", predicates_indexes)
        indx = predicates_indexes[pos] #tf.gather(predicates_indexes, pos)  # Tensor-safe indexing
        name = GlobalData.get_key_from_indx_predicates_list(indx)

        return self.state.lookup(name)


    def extract_from_lifted_old(self, node_name, predicates_indexes):
        pos_str = node_name.replace(LIFTED_STR, "")
        pos=int(pos_str)
        indx= predicates_indexes[pos]
        name=GlobalData.get_key_from_indx_predicates_list(indx)
        return self.state.lookup(name)


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
            sp.Min: tf.minimum,  # ReLU equivalent for Max
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
        #tf.print("Node: ", node)
        return float(node)
        #return tf.constant(float(node), dtype=tf.float32)
    
    def is_tensor(self, node):
        return node is tf.Tensor
    