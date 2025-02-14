from abc import ABC, abstractmethod

from typing import Union, Optional, List, Dict

import tensorflow as tf
import sympy  
import sympy as sp
from enum import Enum
import copy

import sympy 
from sympy import Max, sin, lambdify
from sympy.abc import x as sp_data

import unified_planning as up
from unified_planning.model import OperatorKind
from unified_planning.model import EffectKind
from unified_planning.tensor.constants import *
from unified_planning.engines.compilers import Grounder, GrounderHelper
from unified_planning.tensor.tensor_state import TensorState, TfState
from unified_planning.tensor.tensor_fluent import TensorFluent, TfFluent

from unified_planning.tensor.converter import tensor_convert

class PreconditionData: 
    """Data structure to store all relevant information about a precondition."""
    def __init__(self, prec, position, name, sympy_expr, predicates_keys, predicates_indexes):
        self.prec = prec
        self.position = position
        self.name = name
        self.sympy_expr = sympy_expr
        self.predicates_keys = predicates_keys
        self.predicates_indexes =  predicates_indexes

    def __repr__(self):
        return (f"PreconditionData(position={self.position}, prec={self.prec}, "
                f"name={self.name}, sympy_expr={self.sympy_expr})")

class EffectData:
    """Data structure to store all relevant information about an effect."""
    def __init__(self, effect, position, name, condition, sympy_sat, sympy_unsat, predicates_keys, predicates_indexes):
        self.effect = effect
        self.position = position
        self.name = name
        self.condition = condition
        self.sympy_sat = sympy_sat
        self.sympy_unsat = sympy_unsat
        self.predicates_keys = predicates_keys
        self.predicates_indexes =  predicates_indexes

    def __repr__(self):
        return (f"EffectData(position={self.position}, effect={self.effect}, "
                f"condition={self.condition}, sympy_sat={self.sympy_sat}, "
                f"sympy_unsat={self.sympy_unsat})")


class TfUPAction:
    _class_action_id = 0  # Class attribute to keep track of the class ID

    def __init__(self, up_action, converter):
        self.up_action = up_action
        self.converter = converter

        self._act_effects_list=[]
        self._act_preconditions_list=[]

        self.action_id = TfUPAction._class_action_id  # Assign a unique ID to the action
        TfUPAction._class_action_id += 1

        self._build_preconditions_effects()


    def get_action_id(self):
        return self.action_id


    def store_condition(self, prec: up.model.Precondition):
        """
        Store a precondition in the class list and map if it does not already exist.
        """
        converter= self.converter
        if prec in GlobalData._class_conditions_map:
            cond_position = GlobalData._class_conditions_map[prec]
            if DEBUG > 5:
                print(f"Precondition {prec} already exists at position {cond_position}")
        else:
            fl_name = prec.get_name()
            prec_str= converter.get_condition_str(prec)

            fve = up.model.walkers.FreeVarsExtractor()
            predicates_set= fve.get(prec)
            predicates_indexes=GlobalData.insert_predicates_in_map(predicates_set)
            #predicates_keys=GlobalData.get_keys_from_predicates_map(predicates_indexs)

            lifted_prec_str=GlobalData.get_lifted_string(prec_str,predicates_set)
            sympy_expr =  converter.define_condition_expr(lifted_prec_str) 

            cond_position = len(GlobalData._class_conditions_list)
            # Create a PreconditionData object
            prec_data = PreconditionData(
                prec=prec,
                position=cond_position,
                name=fl_name,
                sympy_expr=sympy_expr,
                predicates_keys=predicates_set,
                predicates_indexes=predicates_indexes
            )
            # Store in list and map
            GlobalData._class_conditions_list.insert(cond_position, prec_data)
            GlobalData._class_conditions_map[prec] = cond_position
            if DEBUG > 5:
                print(f"Added new precondition: {prec_data}")
        return cond_position
    

    def store_effect(self, effect: up.model.Effect):
        """
        Store an effect in the class list and map if it does not already exist.
        """
        converter= self.converter           
        if DEBUG>5:
            print("Define effect:", effect)

        if effect in GlobalData._class_effects_map:

            ef_position = GlobalData._class_effects_map[effect]
            if DEBUG > 5:
                print(f"Effect {effect} already exists at position {ef_position}")
        else:
            ef_position = len(GlobalData._class_effects_list)
            fl_name = effect.fluent.get_name()  # Assuming fluent() provides the name
            ef_str= str(effect.value)

            fve = up.model.walkers.FreeVarsExtractor()
            effect_set= fve.get(effect.fluent)| fve.get(effect.value) |fve.get(effect.condition)
  
            effect_indexes=GlobalData.insert_predicates_in_map(effect_set)

            # Compute sympy expressions
            sympy_expr_sat = converter.when_sympy_expr_not_inserted(effect, effect_set, 1.0)
            sympy_expr_unsat = converter.when_sympy_expr_not_inserted(effect, effect_set, -1.0)
            cond_position=-1 #No conditional effect
            if effect.is_conditional():
                cond_position=TensorAction.store_condition(effect.condition, converter)
                

            # Create an EffectData object
            effect_data = EffectData(
                effect=effect,
                position=ef_position,
                name=fl_name,
                condition=cond_position,
                sympy_sat=sympy_expr_sat,
                sympy_unsat=sympy_expr_unsat,
                predicates_keys=effect_set,
                predicates_indexes=effect_indexes
            )

            # Store in list and map
            GlobalData._class_effects_list.append(effect_data)
            GlobalData._class_effects_map[effect] = ef_position

            if DEBUG > 5:
                print(f"Added new effect: {effect_data}")
            
        return ef_position

    def evaluate_condition(self, condition: int):
        """
        Evaluates a single precondition and returns the resulting value.
        
        Args:
            precondition (Precondition): A precondition to evaluate.
        
        Returns:
            Tensor: The evaluation result of the precondition.
        """
        converter= self.converter
        sympy_expr = GlobalData._class_conditions_list[condition].sympy_expr
        predicates_indexes=GlobalData._class_conditions_list[condition].predicates_indexes
        value= converter.convert(sympy_expr, predicates_indexes)
        
        return value 


    def _build_preconditions_effects(self):
        """
        Apply the effects of the action to the new state.

        Args:
            effects (list): A list of effect objects to apply.
            new_state (dict): The new state that will be modified.
        """
        #PRECONDITIONS
        preconditions = self._up_action.preconditions
        for prec in preconditions:
            if DEBUG>5:
                print("Define precondition:", prec)
            cond_position = TensorAction.store_condition(prec, self.converter)
            # Store the precondition position to build the action's preconditions_list
            self._act_preconditions_list.append(cond_position)
            
        self._act_preconditions_list=self._act_preconditions_list

        #EFFECTS
        effects = self._up_grounded_action.effects
        for effect in effects:
            ef_position= TensorAction.store_effect(effect, self.converter)
            # Store the effect position to build the action's effects_list
            self._act_effects_list.append(ef_position)



    def __repr__(self):
        return (f"TfUPAction(action_id={self.action_id}, "
                f"up_action={self.up_action}, preconditions={self.preconditions}, "
                f"effects={self.effects})")



# TensorAction class modified to use the SympyToTensorConverter
class TensorAction(ABC):
    _class_action_id = 0  # Class attribute to keep track of the class ID

    def __init__(self, problem: up.model.Problem, plan_action: up.plans.plan.ActionInstance, converter, state):
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
        self._up_grounded_action = grounded_action # Store the action
        self._up_action=plan_action._action
        self._up_params=plan_action.actual_parameters
        if self._up_action in GlobalData._class_up_actions_map:
            self._up_action_id = GlobalData._class_up_actions_map[self._up_action]
        else:
            self._up_action_id=len(GlobalData._class_up_actions_map)
            GlobalData._class_up_actions_map[self._up_action]=self._up_action_id

        self._up_act_subs: Dict[up.model.Expression, up.model.Expression] = dict(
                    zip(plan_action.action.parameters, list(plan_action.actual_parameters))
                )
        fve = up.model.walkers.FreeVarsExtractor()
        predicates_set= set()
        for prec in plan_action.action.preconditions:
            predicates_set.update(fve.get(prec))
        for effect in plan_action.action.effects:
            predicates_set.update(fve.get(effect.fluent))
            predicates_set.update(fve.get(effect.value))
            if effect.is_conditional():
                predicates_set.update(fve.get(effect.condition))
        pred_matchings = dict()
        pred_matchings_up_data = dict()
        for pred in predicates_set:
            if pred.node_type != OperatorKind.FLUENT_EXP :
                print("Error: Not a fluent expression: ", pred)
                continue
            new_pred = pred.substitute(self._up_act_subs)
            pred_matchings.update({str(pred): str(new_pred)})
            pred_matchings_up_data.update({pred: new_pred})

        self._up_act_pred_matchings_up_data = pred_matchings_up_data
        self._up_act_pred_matchings = pred_matchings
        self._up_act_free_vars = predicates_set
        self.curr_state = None
        self.new_state = None
        self._are_preconditions_satisfied=0
        self.converter=converter

        if state is not None:
            self.curr_state = state
            self.converter.set_state(self.curr_state)

        self._action_id = TensorAction._class_action_id  # Assign a unique ID to the action
        TensorAction._class_action_id += 1  # Increment the class ID for the next action
        if DEBUG>1:
            print("Action ID:", self._action_id)
        
        self._tf_up_action = TfUPAction(self._up_action, self.converter)

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
        return self._up_grounded_action.name
    
    def get_up_action_id(self):
        return self._up_action_id
   
    def apply_action(self, curr_state): # up_id useful for tf.function
        """
        Apply an action to the state if the preconditions are met, and manage the effects.
        
        Args:
            curr_state (dict): The initial state as a dictionary of fluent names to values.
        
        Returns:
            cost (Tensor): The cost calculated by applying the action's effects, or an invalid state if preconditions are not met.
        """
        if curr_state is not None:
            self.curr_state=curr_state

        self.converter.set_state(self.curr_state)
        
        if DEBUG>6:
            print("Initial state:", self.curr_state)
            #tf.print(".Initial state:", curr_state['objective'])

        # Evaluate preconditions
        self._are_preconditions_satisfied = self.evaluate_preconditions(up_action_id)
        if DEBUG>1:
            if self._are_preconditions_satisfied<0:    
                print("Preconditions not satisfied, action id: ", self._action_id, " name: ", self.get_name())
        # Apply effects if preconditions are satisfied
        state_update=self.apply_effects(up_action_id)

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
        if self._are_preconditions_satisfied<0:
            metric_value =  metric_value + UNSAT_PENALTY  #  * self._are_preconditions_satisfied)
     
        state_update[str(metric.expression)]=metric_value

        if DEBUG>0:
            if self._are_preconditions_satisfied<0:    
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
        return self._are_preconditions_satisfied>=0
    


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
                f"action={self._up_grounded_action}, "
                f"is applicable={self._are_preconditions_satisfied}, "
                #f"state={self.new_state}) "
                ) 
    

class TfAction(TensorAction):
    def __init__(self, problem: up.model.Problem, plan_action: up.plans.plan.ActionInstance, converter, state):
        """
        Initialize the TensorAction with the problem, action, and converter.
        
        Args:
            problem (Problem): The problem containing the objective and other parameters.
            action (Action): The action to apply.
        """
        super().__init__(problem, plan_action, converter, state)
 
    #@tf.function #(reduce_retracing=True)
    def apply_action(self,curr_state=None): #: up.tensor.TensorState):    
        """
        Apply TensorfFlow  action to the state if the preconditions are met, and manage the effects.
        """
        new_state=super().apply_action(curr_state)
        return new_state


    def no_tf_apply_action(self, curr_state=None): #: up.tensor.TensorState):    
        """
        Apply TensorfFlow  action to the state if the preconditions are met, and manage the effects.
        """
        new_state=super().apply_action(curr_state)
        return new_state

    def evaluate_preconditions(self, up_action_id,  state=None):
        """
        Evaluates all preconditions and returns whether they are satisfied.
        
        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """
        if DEBUG>5:
            print("Evaluate preconditions, act: ", self.get_name(), " id:", up_action_id)

        if state is not None:
            self.curr_state = state
            self.converter.set_state(state)
            
        preconditions = self._act_preconditions_list
        satisfied = tf.constant(0.0)  # Assume preconditions are satisfied initially; negative value indicates unsatisfied

        for prec in preconditions:
            if DEBUG>2:            
                print("Evaluating precondition:", prec)
            value = TensorAction.evaluate_condition(prec, self.converter)
            result = tf.minimum(value, 0.0) # 0.0 if the precondition is satisfied, negative otherwise
            satisfied = tf.add(satisfied, result)

            if DEBUG>0:
                if result >= 0:
                    if DEBUG>2:
                        print("Precondition satisfied: ", prec)
                else:
                    print("Precondition not satisfied: ", GlobalData._class_conditions_list[prec]," indx:", prec ," value: ", value, " act: ", self.get_name(), " id:", self.get_action_id())

        # Check if all preconditions are satisfied
        return satisfied

    

    #@tf.function
    def apply_effects(self, up_id): #up_id useul for tf.function retracing
        """
        Apply the effects of the action to the new state.

        Args:
            effects (list): A list of effect objects to apply.
            new_state (dict): The new state that will be modified.
        """
        if DEBUG>5:
            print("Apply effects, act: ", self.get_name(), " id:", up_id)
        effects = self._act_effects_list
        #self.converter = self.converter_funct(curr_state, self)#  converter (SympyToTensorConverter): The converter instance for SymPy to TensorFlow conversion.
        out_result={}
        for effect_indx in effects:
            effect_data=GlobalData._class_effects_list[effect_indx]
            if DEBUG>5:
                print("Applying effect:", effect_data.effect)
            result=1.0
            if effect_data.condition>=0:
                result = TensorAction.evaluate_condition(effect_data.condition, self.converter) 
                if DEBUG>4:
                    print("Condition: ", effect_data.condition)
                    print("Condition result:", result)
            if result>0:
                out_result.update(self._apply_single_effect(effect_indx))
        
        return out_result

    #@tf.function
    def _apply_single_effect(self, effect_indx):
        """
        Apply a single effect to the new state.

        Args:
            effect (Effect): The effect to apply.
            curr_state (dict): The initial state.
        """
        effect_data = GlobalData._class_effects_list[effect_indx]
        fl_name = effect_data.name
        fl_var_name = fl_name.replace('(', '_').replace(')', '_')
       
        if DEBUG>4:
            print("Effect:", effect_data.effect)
        result=0.0

        if(DEBUG>2):
            print("..compute_effect_value: ", fl_name, " - indx: ", effect_indx)
        
        are_prec_satisfied=1.0 #self._are_preconditions_satisfied
        sympy_expr = effect_data.sympy_sat
        result=0.0
        if (are_prec_satisfied < 0.0):
            sympy_expr = effect_data.sympy_unsat

        result= self.converter.convert(sympy_expr, effect_data.predicates_indexes ,  are_prec_satisfied)
        #result= tensorconverter(sympy_expr, are_prec_satisfied, self.curr_state)
        if(DEBUG>5):
            print("Result: ", result)
            tf.print("TResult: ", result)

        if DEBUG>4:
            value=float(self.curr_state[fl_name])
               
            print("-->",fl_name,"- init:",value, " delta:", result-value, " new: ", result)
            #self.new_state[fl_name]=(result) #TfFluent( effect.fluent,result)

        return {fl_name: result }
