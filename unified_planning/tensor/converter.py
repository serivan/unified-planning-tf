
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



class SympyToTensorConverter(ABC):
    """
    A utility class to handle the conversion of SymPy expressions to TensorFlow operations.
    """
    _class_current_state=None


    def __init__(self, state: up.model.State): #check use:: tensor_action:unified_planning.tensor.TensorAction
        self.tensor_state = state  # The state provides the values for the SymPy variables.
        SympyToTensorConverter._class_current_state=state
    

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
        #result= self._tensor_convert(sympy_expr,predicates_indexes, are_prec_satisfied)
        result= SympyToTensorConverter.tensor_convert(sympy_expr,predicates_indexes, self.tensor_state.get_state_values(), are_prec_satisfied)
        if(DEBUG>4):
            print("Converter Result:", result)
        return result
    
    # Wrap the conversion into a tf.function
    def create_tensor_function(sympy_expr, predicates_indexes, are_prec_satisfied=tf.constant(1.0)):
        #@tf.function
        def tensor_convert_funct(state):
            return SympyToTfConverter.tensor_convert(sympy_expr, predicates_indexes, state, are_prec_satisfied)
        return tensor_convert_funct


    # Wrap the conversion into a tf.function
    def sympy_to_tensor_function_conv(sympy_expr, are_prec_satisfied=tf.constant(1.0)):
        #@tf.function
        def tensor_convert_funct(predicates_indexes, state_values):
            return SympyToTfConverter.tensor_convert(sympy_expr, predicates_indexes, state_values, are_prec_satisfied)
        return tensor_convert_funct

    def sympy_to_tensor_function(sympy_expr, are_prec_satisfied=tf.constant(1.0)):
        # Convert the string to a Sympy expression.
        #sympy_expr = sympy.sympify(expr)
        
        # Extract free symbols from the expression and sort them by name.
        free_symbols = sorted(sympy_expr.free_symbols, key=lambda s: s.name)
        arg_names = [str(s) for s in free_symbols]
        arg_indexes = [int(name.split('_')[-1]) for name in arg_names]
        # Create a lambda function using sympy.lambdify with TensorFlow as backend.
        need_lambdify=False
        
        if isinstance(sympy_expr, sympy.logic.boolalg.BooleanTrue):
            f_lambdified = lambda : TF_SAT
        elif isinstance(sympy_expr, sympy.logic.boolalg.BooleanFalse):
            f_lambdified = lambda : TF_UN_SAT
        elif isinstance(sympy_expr,  sympy.Number):
            f_lambdified = lambda : sympy_expr if SympyToTfConverter.is_tensor(sympy_expr) else SympyToTfConverter.get_constant(sympy_expr)
        else:
            f_lambdified = sympy.lambdify(free_symbols, sympy_expr, modules="tensorflow")
            need_lambdify=True
        
        #@tf.function
        def tf_func(indexes,state_values):
            # Ensure that the number of indexes matches the number of free symbols.
            #if len(indexes) < len(free_symbols):
            #    raise ValueError("The number of indexes must equal the number of free symbols in the expression.")
        
            # Extract selected tensor elements using the provided indexes.
            #selected =[tensor[indexes[i]] for i in arg_indexes]  # [tensor[i] for i in arg_indexes
                 
            if need_lambdify:
                selected= tf.unstack(tf.gather(state_values, tf.gather(indexes, arg_indexes)))
                result=f_lambdified(*selected)
            else:
                result=f_lambdified()
            if DEBUG>3:
                if len(arg_indexes)<=0:
                    new_indexes=[]
                else:
                    new_indexes= tf.gather(indexes, arg_indexes)
                print("..Lambdify: ", sympy_expr, " indexes: ", new_indexes)
                keys=[ GlobalData._class_tensor_state.get_key(i) for i in new_indexes]
                
                if need_lambdify:
                    print("Selected: ", selected)
                print("Keys: ", keys)

            # Evaluate the expression using the lambdified function.
            return result
        
        return tf_func


    def convert_new(sympy_expr,predicates_indexes, are_prec_satisfied=0.0):
        """
        Convert a SymPy expression to a TensorFlow tensor by recursively evaluating it.

        Args:
            sympy_expr (sympy.Expr): The SymPy expression to convert.

        Returns:
            TensorFlow Tensor: The resulting TensorFlow tensor.
        """
        if(DEBUG>4):
            print("..convert")

        #result= SympyToTensorConverter.tensor_convert(sympy_expr,predicates_indexes, state, are_prec_satisfied)
        result= SympyToTensorConverter.tensor_convert_it(sympy_expr,predicates_indexes, are_prec_satisfied)
        #result= tensor_convert(sympy_expr, are_prec_satisfied, self.state)

        if(DEBUG>4):
            print("Converter Result:", result)
        return result


    def set_state(self, state):
        ''' Set the state to be used for conversion
            Args: state (State): The state to use
        ''' 
        self.state = state
        SympyToTensorConverter._class_current_state=state


    def get_curr_state():
        '''
        Get the current state
        '''
        return SympyToTensorConverter._class_current_state

    def when_sympy_expr_inserted(self, effect):
        return effect.sympy_expr

    def when_sympy_expr_not_inserted(self, effect, effects_set, are_prec_satisfied):
        # Handle different effect kinds
        if  effect.fluent.node_type == OperatorKind.FLUENT_EXP and effect.value.is_bool_constant():
            if effect.value.bool_constant_value():
                value_str = str(NUM_SAT)
            else:
                value_str = str(NUM_UN_SAT)
            value_str = str(NUM_UN_SAT) if are_prec_satisfied < 0.0 else value_str+'-'+str(effect.fluent.get_name())
        elif effect.kind == EffectKind.ASSIGN:
            value_str = "0.0" if are_prec_satisfied < 0.0 else str(effect.value) + '-' + str(effect.fluent.get_name())

        elif effect.kind == EffectKind.INCREASE:
            value_str = "0.0" if are_prec_satisfied < 0.0 else  str(effect.value)

        elif effect.kind == EffectKind.DECREASE:
            value_str = "0.0" if are_prec_satisfied < 0.0 else  '(-1.0*)' + str(effect.value)
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
        
        if condition.node_type == OperatorKind.NOT or condition.node_type == OperatorKind.AND or condition.node_type == OperatorKind.OR: #Conditions already handled
            value_str = self.define_condition_str(condition)
        else:
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
    def tensor_convert(node, predicates_indexes, state, are_prec_satisfied=tf.constant(1.0)):
        """
        Recursively convert a SymPy node into a TensorFlow operation.
        
        Args:
            node (sympy.Basic): A SymPy expression node to convert.
        
        Returns:
            TensorFlow Tensor: The resulting tensor.
        """
        if DEBUG>5:
            print("..t_convert: ", node, " predicates_indexes: ", predicates_indexes)
    
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
                value=SympyToTfConverter.extract_from_lifted(lookup_key, predicates_indexes)
            elif tf.not_equal(state.lookup(lookup_key), MISSING_VALUE): 
                if DEBUG > 5:
                    print("Node:", node_name, "in state:", state[node_name])
                value = state.lookup(lookup_key)
            else:
                value = tf.constant(-1.0)
        elif isinstance(node, sympy.Number):
            value = node if SympyToTfConverter.is_tensor(node) else SympyToTfConverter.get_constant(node)
        elif isinstance(node, sympy.Basic):
            if isinstance(node, sympy.Add):
                args = [SympyToTfConverter.tensor_convert(arg, predicates_indexes, state,  are_prec_satisfied) for arg in node.args]
                value = tf.add_n(args)
            elif isinstance(node, sympy.Mul):
                args = [SympyToTfConverter.tensor_convert(arg, predicates_indexes, state, are_prec_satisfied) for arg in node.args]
                value = tf.math.reduce_prod(args)
            elif isinstance(node, sympy.Pow):
                func = SympyToTfConverter.sympy_to_tensor_map.get(type(node))
                base, exp = [SympyToTfConverter.tensor_convert(arg, predicates_indexes, state, are_prec_satisfied) for arg in node.args]
                value = func(base, exp)      
            #elif  tf.not_equal(self.state.lookup(lookup_key), MISSING_VALUE):
            #    if DEBUG > 5:
            #        tf.print("Node:", node_name, "in state:", self.state[node_name])
            #    value = self.state.lookup(lookup_key)
            #elif   tf.equal(tf.strings.substr(lookup_key, 0, len(LIFTED_STR)), LIFTED_STR): #tf.cond( tf.equal(tf.strings.substr(lookup_key, 0, len(LIFTED_STR)), LIFTED_STR), lambda: True, lambda: False): 
            #    value=self.extract_from_lifted(lookup_key, predicates_indexes)    
            else:
                func = SympyToTfConverter.sympy_to_tensor_map.get(type(node))
                if func:
                    value = func(*[SympyToTfConverter.tensor_convert(arg,predicates_indexes, state,  are_prec_satisfied) for arg in node.args])
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
        
        if DEBUG > 5:
            print(str(node), ":=", value)
        
        return value

    #@tf.function #(input_signature=[ tf.TensorSpec(shape=(), dtype=tf.int32),  tf.TensorSpec(shape=[None], dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.float32) Z])  
    def tensor_convert_it(node_condition, predicates_indexes, are_prec_satisfied):
        """
        Iteratively convert a SymPy node into a TensorFlow operation, allowing manual selection of GPU or CPU execution.
        
        DEVICE (str): Device to execute on, e.g., '/GPU:0' or '/CPU:0'.
        Args:
            node (sympy.Basic): A SymPy expression node to convert.
        
        Returns:
            TensorFlow Tensor: The resulting tensor.
        """
        if DEBUG>5:
            print("..t_convert: ", GlobalData._class_cond_effects_list[node_condition].sympy_expr, " predicates_indexes: ", predicates_indexes)
    
        with tf.device(DEVICE):
            node=GlobalData._class_cond_effects_list[node_condition].sympy_expr
            state=SympyToTfConverter.get_curr_state()
            #node = sympy.sympify(node_str)
            stack = [(node, None)]  # Stack to store nodes and their computed values
            results = {}
            missing_value = tf.constant(-1.0)
            true_value = tf.constant(1.0)
            false_value = tf.constant(-1.0)
            
            while stack:
                current, parent = stack.pop()
                
                if isinstance(current, tf.Tensor):
                    results[current] = current
                elif isinstance(current, sympy.logic.boolalg.BooleanTrue):
                    results[current] = true_value
                elif isinstance(current, sympy.logic.boolalg.BooleanFalse):
                    results[current] = false_value
                elif isinstance(current, sympy.Symbol):
                    node_name = current.name
                    lookup_key = tf.constant(node_name, dtype=tf.string)
                    
                    if node_name == ARE_PREC_SATISF_STR:
                        results[current] = are_prec_satisfied
                    elif node_name.startswith(LIFTED_STR):
                        results[current] = SympyToTfConverter.extract_from_lifted(lookup_key, predicates_indexes)
                    else:
                        results[current] = tf.cond(
                            tf.not_equal(state.lookup(lookup_key), missing_value),
                            lambda: state.lookup(lookup_key),
                            lambda: missing_value
                        )
                elif isinstance(current, sympy.Number):
                    results[current] = (current if SympyToTfConverter.is_tensor(current) 
                                        else SympyToTfConverter.get_constant(current))
                elif isinstance(current, sympy.Basic):
                    
                    if current in results:  # Skip if already computed
                        continue
                    
                    args = current.args
                    
                    if all(arg in results for arg in args):
                        if isinstance(current, sympy.Add):
                            results[current] = tf.add_n([results[arg] for arg in args])
                        elif isinstance(current, sympy.Mul):
                            results[current] = tf.math.reduce_prod([results[arg] for arg in args])
                        elif isinstance(current, sympy.Pow):
                            func = SympyToTfConverter.sympy_to_tensor_map.get(type(current))
                            results[current] = func(results[args[0]], results[args[1]])
                        else:
                            func = SympyToTfConverter.sympy_to_tensor_map.get(type(current))
                            if func:
                                results[current] = func(*[results[arg] for arg in args])
                            else:
                                raise ValueError(f"Unsupported operation: {type(current)}")
                    else:
                        stack.append((current, parent))  # Re-add current node to process after children
                        for arg in args:
                            if arg not in results:
                                stack.append((arg, current))
                else:
                    raise ValueError(f"Unsupported node type: {type(current)}")
            
            return results[node]

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
                value=self._extract_from_lifted(lookup_key, predicates_indexes)
            elif tf.not_equal(self.state.lookup(lookup_key), MISSING_VALUE): 
                if DEBUG > 5:
                    print("Node:", node_name, "in state:", self.state[node_name])
                value = self.state.lookup(lookup_key)
            else:
                value = tf.constant(-1.0)
        elif isinstance(node, sympy.Number):
            value = node if self._is_tensor(node) else self._get_constant(node)
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
            #elif  tf.not_equal(self.state.lookup(lookup_key), MISSING_VALUE):
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
            print(str(node), ":=", value)
        
        return value
    
    def _extract_from_lifted(self, node_name: tf.Tensor, predicates_indexes: tf.Tensor):
        pos_str = tf.strings.regex_replace(node_name, LIFTED_STR, "")
        pos = tf.strings.to_number(pos_str, out_type=tf.int32)  # Convert to integer
        indx = predicates_indexes[pos] #tf.gather(predicates_indexes, pos)  # Tensor-safe indexing
        name = GlobalData.get_key_from_indx_predicates_list(indx)
        state=SympyToTfConverter.get_curr_state()

        if DEBUG > 4:
            print("Fluent: ", name," val: ", state.lookup(name))
        if DEBUG > 6:          
            printf(" Pos: ", pos, "Predicates indexes: ", predicates_indexes)
        return self.state.lookup(name)


    def extract_from_lifted(node_name: tf.Tensor, predicates_indexes: tf.Tensor): #XXX This or previous one
        pos_str = tf.strings.regex_replace(node_name, LIFTED_STR, "")
        pos = tf.strings.to_number(pos_str, out_type=tf.int32)  # Convert to integer
        indx = predicates_indexes[pos] #tf.gather(predicates_indexes, pos)  # Tensor-safe indexing
        name = GlobalData.get_key_from_indx_predicates_list(indx)
        state=SympyToTfConverter.get_curr_state()

        if DEBUG > 4:
            print("Fluent: ", node_name," val: ", state.lookup(name))
        if DEBUG > 6:          
            print(" Pos: ", pos, "Predicates indexes: ", predicates_indexes)
        return state.lookup(name)



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
        
    sympy_to_tensor_map = { #XXX only this or previous
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
    
    def _get_constant(self, node): 
        #tf.print("Node: ", node)
        return float(node)
        #return tf.constant(float(node), dtype=tf.float32)
    
    def _is_tensor(self, node):
        return node is tf.Tensor
    
    
    def get_constant( node): 
        #tf.print("Node: ", node)
        return float(node)
        #return tf.constant(float(node), dtype=tf.float32)
    
    def is_tensor( node):
        return node is tf.Tensor