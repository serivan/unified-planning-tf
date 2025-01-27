from abc import ABC, abstractmethod

from typing import Union, Optional, List, Dict

import tensorflow as tf
import sympy as sp
from enum import Enum
import copy

import sympy 

import unified_planning as up
from unified_planning.model import OperatorKind
from unified_planning.model import EffectKind
from unified_planning.tensor.constants import *
from unified_planning.engines.compilers import Grounder, GrounderHelper
from unified_planning.tensor.tensor_state import TensorState, TfState
from unified_planning.tensor.tensor_fluent import TensorFluent, TfFluent

# TensorAction class modified to use the SympyToTensorConverter
class TensorAction(ABC):
    _global_action_id = 0  # Class attribute to keep track of the global ID

    def __init__(self, problem: up.model.Problem, plan_action: up.plans.plan.ActionInstance, converter_funct):
        """
        Initialize the TensorAction with the problem and action.
        
        Args:
            problem (Problem): The problem containing the objective and other parameters.
            action (Action): The action to apply, which has preconditions and effects.
        """
        self.problem = problem  # The problem containing objective and other parameters
        #pk = problem.kind
        #if not Grounder.supports(pk):
        #    msg = f"The Grounder used in the {type(self).__name__} does not support the given problem"
        #    if self.error_on_failed_checks:
        #        raise UPUsageError(msg)
        #    else:
        #        warn(msg)
        self._grounder = GrounderHelper(problem,prune_actions=False)
        self._plan_action =  plan_action
        grounded_action = self._grounder.ground_action(plan_action._action, plan_action.actual_parameters )
        if grounded_action is None:
            raise UPInvalidActionError("Apply_unsafe got an inapplicable action.")
        self._up_action = grounded_action # Store the action
        self.curr_state = None
        self.new_state = None
        self._are_preconditions_satisfied=0
        self._converter_funct=converter_funct
        
        
        self._action_id = TensorAction._global_action_id  # Assign a unique ID to the action
        TensorAction._global_action_id += 1  # Increment the global ID for the next action
        if DEBUG>1:
            print("Action ID:", self._action_id)


    def set_curr_state(self, state):
        self.curr_state=state
    
    def set_new_state(self, state):
        self.new_state=state

    def get_curr_state(self):
        return self.curr_state

    def get_next_state(self):
        return self.new_state
    
    def get_action_id(self):
        return self._action_id
    
    def get_name(self):
        return self._up_action.name
   
    def apply_action(self, curr_state): #: up.tensor.TensorState):
        """
        Apply an action to the state if the preconditions are met, and manage the effects.
        
        Args:
            curr_state (dict): The initial state as a dictionary of fluent names to values.
        
        Returns:
            cost (Tensor): The cost calculated by applying the action's effects, or an invalid state if preconditions are not met.
        """
        if curr_state is not None:
            self.curr_state=curr_state

        self._converter = self._converter_funct(self.curr_state)
        
        if DEBUG>6:
            print("Initial state:", self.curr_state)
            #tf.print(".Initial state:", curr_state['objective'])

        # Evaluate preconditions
        self._are_preconditions_satisfied = self.evaluate_preconditions()
 
        # Apply effects if preconditions are satisfied
        state_update=self.apply_effects()

        state_update[ARE_PREC_SATISF_STR]=self._are_preconditions_satisfied #UPDATE state_update to export value for tf.function
   
        # Update the metric value if the preconditions are not satisfied
        metric=self.problem.quality_metrics[0]
        metric_expr=str(metric.expression)
        if metric_expr in state_update:
            metric_value = state_update[metric_expr]
        else:
            metric_value = self.curr_state[metric_expr]
        #metric_value = tf.cond(
        #    tf.greater(self._are_preconditions_satisfied, 0.0),
        #    lambda: self.new_state[str(metric.expression)] ,# Default or alternative value
        #    lambda: self.new_state[str(metric.expression)] + 50000.0 
        #)
        if self._are_preconditions_satisfied<=0:
            metric_value =  metric_value + UNSAT_PENALTY  #  * self._are_preconditions_satisfied)
     
        state_update[str(metric.expression)]=metric_value

        if DEBUG>0:
            if self._are_preconditions_satisfied<=0:    
                print("Preconditions not satisfied, action id: ", self._action_id, " name: ", self.get_name())
            else:
                print("Preconditions satisfied, action id: ", self._action_id, " name: ", self.get_name())
            print("Metric value: ", metric_value)
            print()

        if DEBUG>1:
            print("Update after action:", end=":: ")
            TensorState.print_filtered_dict_state(state_update,["next"])
            print()
        return state_update

    def get_next_state(self):
        return self.new_state
    def get_are_preconditions_satisfied (self):
        return self._are_preconditions_satisfied
    def is_applicable (self):
        return self._are_preconditions_satisfied
    


    @abstractmethod
    def evaluate_preconditions(self, state=None):
        """
        Evaluates all preconditions and returns whether they are satisfied.
        
        Args:
            state (dict): The current state of the problem (variables).
            
        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """
        pass



    @abstractmethod
    def apply_effects(self):
        """
        Apply the effects of the action to the new state.

        Args:
            effects (list): A list of effect objects to apply.
            curr_state (dict): The initial state.
            self.new_state (dict): The new state that will be modified.
        """
        pass

    def __repr__(self):
        return (f"TensorAction(action_id={self._action_id}, "
                f"action={self._up_action}, "
                f"is applicable={self._are_preconditions_satisfied}, "
                #f"state={self.new_state}) "
                ) 
    

class TfAction(TensorAction):
    def __init__(self, problem: up.model.Problem, plan_action: up.plans.plan.ActionInstance, converter_funct):
        """
        Initialize the TensorAction with the problem, action, and converter.
        
        Args:
            problem (Problem): The problem containing the objective and other parameters.
            action (Action): The action to apply.
        """
        super().__init__(problem, plan_action,converter_funct)
 
    #@tf.function
    def apply_action(self, curr_state=None): #: up.tensor.TensorState):    
        """
        Apply TensorfFlow  action to the state if the preconditions are met, and manage the effects.
        """
        new_state=super().apply_action(curr_state)
        return new_state


    def evaluate_preconditions(self, state=None):
        """
        Evaluates all preconditions and returns whether they are satisfied.
        
        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """
        if state is not None:
            self.curr_state = state
            self._converter.set_state(state)
            
        preconditions = self._up_action.preconditions
        satisfied = tf.constant(1.0)  # Assume preconditions are satisfied initially

        for prec in preconditions:
            if DEBUG>2:            
                print("Evaluating precondition:", prec)
            result = self._evaluate_single_precondition(prec)
            if isinstance(result, tf.Tensor) or isinstance(result, tf.Variable):
                value = result
            elif isinstance(result, TfFluent):
                value= self.curr_state[result.get_name()]
            else:
                value= result()
            result = tf.keras.activations.relu(tf.cast(value, tf.float32))  # ReLU activation for precondition evaluation
            satisfied = tf.multiply(satisfied, result)

            if DEBUG>0:
                if result > 0:
                    if DEBUG>2:
                        print("Precondition satisfied")
                else:
                    print("Precondition not satisfied: ", prec, " value: ", value, " act: ", self.get_name(), " id:", self.get_action_id())

        # Check if all preconditions are satisfied
        return tf.keras.activations.relu(tf.sign(satisfied))

    def _evaluate_single_precondition(self, precondition: Union[
            "up.model.fnode.FNode",
            "up.model.fluent.Fluent",
            "up.model.parameter.Parameter",
            bool]):
        """
        Evaluates a single precondition and returns the resulting value.
        
        Args:
            precondition (Precondition): A precondition to evaluate.
        
        Returns:
            Tensor: The evaluation result of the precondition.
        """
        if precondition.node_type == OperatorKind.FLUENT_EXP:
            if precondition.get_name() in self.curr_state:
                return (self.curr_state[precondition.get_name()])
            else:
                return tf.constant(0.0) # tf.Variable(0.0, dtype=tf.float32, trainable=False)
        else:
            value= self._converter.compute_condition_value(precondition) 
            #value_str = str(value)  # Convert value to string
            #sympy_expr = sp.sympify(value_str)
            return value #self._converter.convert(sympy_expr)

    #@tf.function
    def apply_effects(self):
        """
        Apply the effects of the action to the new state.

        Args:
            effects (list): A list of effect objects to apply.
            new_state (dict): The new state that will be modified.
        """
        effects = self._up_action.effects
        #self._converter = self._converter_funct(curr_state, self)#  converter (SympyToTensorConverter): The converter instance for SymPy to TensorFlow conversion.
        out_result={}
        for effect in effects:
            if DEBUG>5:
                print("Applying effect:", effect)
            result=1.0
            if effect.is_conditional():
                result = self._evaluate_single_precondition(effect.condition)
                if DEBUG>4:
                    print("Condition: ", effect.condition)
                    print("Condition result:", result)
            if result>0:
                out_result.update(self._apply_single_effect(effect))
        
        return out_result

    #@tf.function
    def _apply_single_effect_old(self, effect):
        """
        Apply a single effect to the new state.

        Args:
            effect: The effect to apply, which includes a fluent and a value.
        Returns:
            A dictionary with the updated fluent value.
        """
        fl_name = effect.fluent.get_name()  # Assuming fluent() provides the name
        fl_var_name = fl_name.replace('(', '_').replace(')', '_')

        result = result1 = tf.constant(0.0, dtype=tf.float32)

        # Handling boolean constant effects
        def handle_boolean_effect():
            #value = tf.cond(
            #    tf.constant(fl_name in self.curr_state, dtype=tf.bool),
            #    lambda: self.curr_state[fl_name],
            #    lambda: tf.constant(0.0, dtype=tf.float32)
            #)
            #are_preconditions_satisfied =  self.get_are_preconditions_satisfied()
            result = tf.constant(effect.value.bool_constant_value(), tf.float32) #tf.multiply(tf.cast(effect.value.bool_constant_value(), tf.float32), are_preconditions_satisfied)
            #result1 = tf.multiply(value, tf.subtract(tf.constant(1.0, dtype=tf.float32), are_preconditions_satisfied))

            #if DEBUG > 4:
            #    tf.print("-->", fl_name, "- init:", value, " delta:", result + result1 - value, " new:", result + result1)

            return result #, result1

        def handle_non_boolean_effect():
            condition = effect.is_sympy_expression_inserted()
            result = self._converter.compute_effect_value(effect, 1) if condition else self._converter.define_effect_value(effect, 1)

            if DEBUG > 4:
                value = self.curr_state.get(fl_name, tf.constant(0.0, dtype=tf.float32))
                tf.print("-->", fl_name, "- init:", value, " delta:", result - value, " new:", result)

            return result #, tf.constant(0.0, dtype=tf.float32)
        # Compute condition dynamically inside the graph
        #condition = tf.logical_and(
        #    effect.fluent.node_type == OperatorKind.FLUENT_EXP,
        #    effect.value.is_bool_constant()
        #)
        condition=  effect.fluent.node_type == OperatorKind.FLUENT_EXP and effect.value.is_bool_constant()
        result = tf.cond(tf.constant(condition), handle_boolean_effect, handle_non_boolean_effect)

        return {fl_name: result}


    #@tf.function
    def _apply_single_effect(self, effect):
        """
        Apply a single effect to the new state.

        Args:
            effect (Effect): The effect to apply.
            curr_state (dict): The initial state.
        """
        fl_name = effect.fluent.get_name()  # Assuming fluent() provides the name
        fl_var_name = fl_name.replace('(', '_').replace(')', '_')
        if DEBUG>4:
            print("Effect:", effect)
        result=result1=0.0

        if effect.fluent.node_type == OperatorKind.FLUENT_EXP and effect.value.is_bool_constant() :
            '''if tf.math.greater(self.get_are_preconditions_satisfied(),tf.constant(0.0)):
                result= effect.value # tf.constant(effect.value.bool_constant_value(), tf.float32, name=f"{fl_var_name}_{self.get_action_id()}")
                self.new_state[fl_name] = TfFluent( effect.fluent,result)'''
            if(fl_name in self.curr_state):
                value=self.curr_state[fl_name]
            else:
                value=tf.constant(0.0) #tf.Variable(0.0, dtype=tf.float32, trainable=False) #TF_ZERO

            are_preconditions_satisfied=1.0 #self.get_are_preconditions_satisfied()
            result= tf.multiply(tf.cast(effect.value.bool_constant_value(), tf.float32 ), are_preconditions_satisfied) 
            result1=tf.multiply(value, (1.0- are_preconditions_satisfied))
            #result1=tf.multiply(self.new_state[fl_name].get_value(), (1.0-self.get_are_preconditions_satisfied())) # tf.constant(effect.value.bool_constant_value(), tf.float32, name=f"{fl_var_name}_{self.get_action_id()}")

            if DEBUG>4:
                print("-->",fl_name,"- init:",value, " delta:", result+result1-value, " new: ", result+result1)
                
            #self.new_state[fl_name]=(result+result1) #TfFluent( effect.fluent,result+result1)
                
        else:
            if (effect.is_sympy_expression_inserted()):
                result =self._converter.compute_effect_value(effect, 1) 
            else:
                result =self._converter.define_effect_value(effect, 1)
            if DEBUG>4:
                value=0.0
                if fl_name in self.curr_state:
                    value=self.curr_state[fl_name]
                
                print("-->",fl_name,"- init:",value, " delta:", result-value, " new: ", result)
            #self.new_state[fl_name]=(result) #TfFluent( effect.fluent,result)

        out=(result+result1)
        return {fl_name: out }
