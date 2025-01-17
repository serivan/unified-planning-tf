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
        self._grounder = GrounderHelper(problem)
        self._plan_action =  plan_action
        grounded_action = self._grounder.ground_action(plan_action._action, plan_action.actual_parameters)
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
        self.curr_state=curr_state
        self._converter = self._converter_funct(curr_state)
        new_state = TensorState.shallow_copy_dict_state(curr_state)
        self.new_state = new_state 
            
        if DEBUG>6:
            print("Initial state:", curr_state)
            #tf.print(".Initial state:", curr_state['objective'])

        # Evaluate preconditions
        self._are_preconditions_satisfied = self.evaluate_preconditions()
        #curr_state[ARE_PREC_SATISF_STR].assign(self._are_preconditions_satisfied)
        #new_state[ARE_PREC_SATISF_STR]=self._are_preconditions_satisfied #UPDATE temporarily also in new_state since the converter works on new_state


        if self._are_preconditions_satisfied<=0:
            if DEBUG>0:
                print("Preconditions not satisfied, action id: ", self._action_id)


        # Apply effects if preconditions are satisfied
        self.apply_effects()
        if DEBUG>5:
            print("State after action:", end=":: ")
            TensorState.print_filtered_dict_state(self.new_state,["next"])
        # Assuming the objective function is defined in the problem
        #objective = self.problem.objective
        #objective_str = str(objective)  # Convert objective to string
        #sympy_expr = sp.sympify(objective_str)
        #cost = converter.convert(sympy_expr)
        return self.new_state

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
    def apply_action(self, curr_state): #: up.tensor.TensorState):    
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
                return tf.Variable(0.0, dtype=tf.float32, trainable=False)
            #return (self.curr_state[precondition.get_name()]).get_value()
        else:
            value= self._converter.compute_condition_value(precondition) 
            #value_str = str(value)  # Convert value to string
            #sympy_expr = sp.sympify(value_str)
            return value #self._converter.convert(sympy_expr)


    def apply_effects(self):
        """
        Apply the effects of the action to the new state.

        Args:
            effects (list): A list of effect objects to apply.
            new_state (dict): The new state that will be modified.
        """
        effects = self._up_action.effects
        #self._converter = self._converter_funct(curr_state, self)#  converter (SympyToTensorConverter): The converter instance for SymPy to TensorFlow conversion.
   
        for effect in effects:
            if DEBUG>7:
                print("Applying effect:", effect)
            result=1.0
            if effect.is_conditional():
                result = self._evaluate_single_precondition(effect.condition)
                if DEBUG>4:
                    print("Condition: ", effect.condition)
                    print("Condition result:", result)
            if result>0:
                self._apply_single_effect(effect)



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

        if effect.fluent.node_type == OperatorKind.FLUENT_EXP and effect.value.is_bool_constant() :
            '''if tf.math.greater(self.get_are_preconditions_satisfied(),tf.constant(0.0)):
                result= effect.value # tf.constant(effect.value.bool_constant_value(), tf.float32, name=f"{fl_var_name}_{self.get_action_id()}")
                self.new_state[fl_name] = TfFluent( effect.fluent,result)'''
            if(fl_name in self.curr_state):
                value=self.curr_state[fl_name]
            else:
                value=tf.constant(0.0) #tf.Variable(0.0, dtype=tf.float32, trainable=False) #TF_ZERO

            result= tf.multiply(tf.cast(effect.value.bool_constant_value(), tf.float32 ), self.get_are_preconditions_satisfied()) 
            result1=tf.multiply(value, (1.0-self.get_are_preconditions_satisfied()))
            #result1=tf.multiply(self.new_state[fl_name].get_value(), (1.0-self.get_are_preconditions_satisfied())) # tf.constant(effect.value.bool_constant_value(), tf.float32, name=f"{fl_var_name}_{self.get_action_id()}")

            if DEBUG>4:
                print("-->",fl_name,"- init:",value, " delta:", result+result1-value, " new: ", result+result1)
                
            self.new_state[fl_name]=(result+result1) #TfFluent( effect.fluent,result+result1)
                
        else:
            if (effect.is_sympy_expression_inserted()):
                result =self._converter.compute_effect_value(effect, 1) 
            else:
                result =self._converter.build_compute_effect_value(effect, 1)
            if DEBUG>4:
                print("-->",fl_name,"- init:",self.new_state[fl_name], " delta:", result-self.new_state[fl_name], " new: ", result)
            self.new_state[fl_name]=(result) #TfFluent( effect.fluent,result)

        if DEBUG>7:
            print("State after effect:", end=":: ")
            TensorState.print_filtered_dict_state(self.new_state,["next"])
