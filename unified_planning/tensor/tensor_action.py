from abc import ABC, abstractmethod

from typing import Union, Optional, List, Dict

import tensorflow as tf
from tensorflow.math import tanh

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

from unified_planning.engines.compilers import Grounder, GrounderHelper
from unified_planning.tensor.tensor_state import TensorState, TfState
from unified_planning.tensor.tensor_fluent import TensorFluent, TfFluent


from unified_planning.tensor.constants import *
from unified_planning.tensor.converter import SympyToTfConverter

from tensorflow.lookup.experimental import MutableHashTable

class LiftedActionData:

    def __init__(self,  lifted_action, 
                lifted_indx, 
                predicates_list, 
                predicates_indexes, 
                act_preconditions_list, 
                act_preconditions_function, 
                act_effects_list,
                act_effect_pos_list,
                apply_act_funct):
        self.lifted_action=lifted_action
        self._lifted_action_indx=lifted_indx
        self.predicates_list=predicates_list
        self.predicates_indexes=predicates_indexes
        self.act_preconditions_list=act_preconditions_list
        self.act_preconditions_function=act_preconditions_function
        self.act_effects_list=act_effects_list
        self.act_effect_pos_list=act_effect_pos_list
        self.apply_action_funct=apply_act_funct
        self.apply_action_concrete_funct=None

    def set_concrete_funct(self, apply_action_concrete_funct):
        self.apply_action_concrete_funct=apply_action_concrete_funct

    def __repr__(self): 
        return (f"LiftedActionData(lifted_action={self.lifted_action}, "
                f"lifted_id={self.lifted_id}, predicates_list={self.predicates_list}, "
                f"predicates_indexes={self.predicates_indexes}, act_preconditions_list={self.act_preconditions_list}, "
                f"act_effects_list={self.act_effects_list})")

class PreconditionData: 
    """Data structure to store all relevant information about a precondition."""
    def __init__(self, prec, position, name, sympy_expr, sympy_function, predicates_keys, predicates_indexes):
        self.prec = prec
        self.position = position
        self.name = name
        self.sympy_expr = sympy_expr
        self.sympy_function=sympy_function
        self.predicates_keys = predicates_keys
        self.predicates_indexes =  predicates_indexes

    def __repr__(self):
        return (f"PreconditionData(position={self.position}, prec={self.prec}, "
                f"name={self.name}, sympy_expr={self.sympy_expr})")

class EffectData:
    """Data structure to store all relevant information about an effect."""
    def __init__(self, effect, position, effect_predicates_position, name, condition, sympy_sat, sympy_sat_function, sympy_unsat,sympy_unsat_function, predicates_keys, predicates_indexes):
        self.effect = effect
        self.position = position
        self.effect_predicates_position=effect_predicates_position
        self.name = name
        self.condition = condition
        self.sympy_sat = sympy_sat
        self.sympy_sat_function=sympy_sat_function
        self.sympy_unsat = sympy_unsat
        self.sympy_unstat_function=sympy_unsat_function
        self.predicates_keys = predicates_keys
        self.predicates_indexes =  predicates_indexes

    def __repr__(self):
        return (f"EffectData(position={self.position}, effect={self.effect}, "
                f"condition={self.condition}, sympy_sat={self.sympy_sat}, "
                f"sympy_unsat={self.sympy_unsat})")


# TensorAction class modified to use the SympyToTfConverter
class TensorAction(ABC):
    _class_action_id = 0  # Class attribute to keep track of the class ID
    
    def __init__(self, problem: up.model.Problem, plan_action: up.plans.plan.ActionInstance, converter, tensor_state):
        """
        Initialize the TensorAction with the problem and action.
        
        Args:
            problem (Problem): The problem containing the objective and other parameters.
            action (Action): The action to apply, which has preconditions and effects.
        """
        self.problem = problem  # The problem containing objective and other parameters
        self._action=plan_action._action
        self.tensor_state=tensor_state
        self._state_values=None
        self._plan_action = plan_action
        self._action_id = TensorAction._class_action_id  # Assign a unique ID to the action
        TensorAction._class_action_id += 1  # Increment the class ID for the next action
        self._act_effects_lifted_list=[]
        self._act_preconditions_list=[]
        self._predicates_list=None
        self._predicates_indexes=None
        self._state_predicates_indexes=None

        self._are_preconditions_satisfied=0
        self.converter=converter
        if tensor_state is not None:
            self.converter.set_state(self.tensor_state)


    def set_tensor_state(self, state):
        self.tensor_state=state
    
    def set_state_values(self, state_values):
        self._state_values=state_values

    def get_tensor_state(self):
        return self.tensor_state

    def get_state_values(self):
        return self._state_values
    
    def get_action_id(self):
        return self._action_id
    
    def get_name(self):
        return self._action.name
    
    def get_predicates_list(self):
        return self._predicates_list
    

    def store_condition(prec, converter, predicates_list=None): # prec: up.model.Precondition): CHECK
        """
        Store a precondition in the class list and map if it does not already exist.
        """

        if prec in GlobalData._class_conditions_map:
            cond_position = GlobalData._class_conditions_map[prec]
            if DEBUG > 5:
                print(f"Precondition {prec} already exists at position {cond_position}")
        else:
            fl_name = prec.get_name()
            prec_str= converter.get_condition_str(prec)
            fve = up.model.walkers.FreeVarsExtractor()
            predicates_set= fve.get(prec)
          
            predicates_indexes=state_predicates_indexes=tf.constant([converter.tensor_state.get_key_position(str(pred)) for pred in predicates_set])
            #sorted(predicates_set, key=len, reverse=True)
            if predicates_list is None:                 
                predicates_list=list(predicates_set)
            else:
                indexes=GlobalData.get_values_from_predicates_list(predicates_list)
                predicates_indexes=tf.constant([t.numpy() for t in indexes], dtype=tf.int32)

            lifted_prec_str=GlobalData.get_lifted_string(prec_str,predicates_list)
            sympy_expr =  converter.define_condition_expr(lifted_prec_str) 
            sympy_function=SympyToTfConverter.sympy_to_tensor_function(sympy_expr)
            cond_position = len(GlobalData._class_conditions_list)
            # Create a PreconditionData object
            prec_data = PreconditionData(
                prec=prec,
                position=cond_position,
                name=fl_name,
                sympy_expr=sympy_expr,
                sympy_function=sympy_function,
                predicates_keys=predicates_list,
                predicates_indexes=predicates_indexes
            )
            # Store in list and map
            GlobalData._class_conditions_list.insert(cond_position, prec_data)
            GlobalData._class_conditions_map[prec] = cond_position
            if DEBUG > 5:
                print(f"Added new precondition: {prec_data}")
        return cond_position
    

    def evaluate_condition(condition: int, predicates_indexes, state_values):
        """
        Evaluates a single precondition and returns the resulting value.
        
        Args:
            precondition (Precondition): A precondition to evaluate.
        
        Returns:
            Tensor: The evaluation result of the precondition.
        """
        #sympy_expr = GlobalData._class_conditions_list[condition].sympy_expr
        #value=  SympyToTfConverter.convert_new(sympy_expr, predicates_indexes, state)
        
        sympy_function = GlobalData._class_conditions_list[condition].sympy_function
        value=  sympy_function(predicates_indexes,state_values)
        return value 


    def _evaluate_condition(self, condition: int, predicates_indexes, state_values):
        """
        Evaluates a single precondition and returns the resulting value.
        
        Args:
            precondition (Precondition): A precondition to evaluate.
        
        Returns:
            Tensor: The evaluation result of the precondition.
        """

        #sympy_expr = GlobalData._class_conditions_list[condition].sympy_expr
        #value=  SympyToTfConverter.convert_new(sympy_expr, predicates_indexes, state)
        
        sympy_function = GlobalData._class_conditions_list[condition].sympy_function
        value=  sympy_function(predicates_indexes,state_values)
        return value 

    def get_next_state(self):
        return self.new_state
    def get_are_preconditions_satisfied (self):
        return self._are_preconditions_satisfied
    
    def set_are_preconditions_satisfied (self, value):
        self._are_preconditions_satisfied=value

    def is_applicable (self):
        return self._are_preconditions_satisfied>=0
    

    def apply_action(self, predicates_indexes, state_values): #: up.tensor.TensorState):
        """
        Apply an action to the state if the preconditions are met, and manage the effects.
        
        Args:
            curr_state (dict): The initial state as a dictionary of fluent names to values.
        
        Returns:
            cost (Tensor): The cost calculated by applying the action's effects, or an invalid state if preconditions are not met.
        """

        # Evaluate preconditions
        self._are_preconditions_satisfied = self.evaluate_preconditions( predicates_indexes, state_values)
        #self._are_preconditions_satisfied = self._act_preconditions_function(predicates_indexes, state_values)
            
        # Apply effects if preconditions are satisfied
        state_update=self.apply_effects(predicates_indexes, state_values)

        state_update.append((self.tensor_state.pos_are_prec_sat, self._are_preconditions_satisfied -  state_values[self.tensor_state.pos_are_prec_sat])) #UPDATE state_update to export value for tf.function

        if DEBUG>0:
            if self._are_preconditions_satisfied<0:    
                print("Preconditions not satisfied, action id: ", self._action_id, " name: ", self.get_name())
            else:
                print("Preconditions satisfied, action id: ", self._action_id, " name: ", self.get_name())
            
            print()

            if DEBUG>1:
                print("Update after action:", end=":: ")
                #TensorState.print_filtered_dict_state(state_update,["next"])
                print(state_update)

        return  self._are_preconditions_satisfied, state_update
    
    
    def _apply_single_effect_new(self, effect_indx, predicates_indexes, state_values):
        """
        Apply a single effect to the new state.

        Args:
            effect (Effect): The effect to apply.
            curr_state (dict): The initial state.
        """

        effect_data = GlobalData._class_effects_list[effect_indx]
        fluent_indx=predicates_indexes[effect_data.effect_predicates_position]

        if(DEBUG>2):
            print("...Apply single effect: ", effect_indx, " predicates: ", predicates_indexes)

            fluent_name =GlobalData._class_tensor_state.get_key(fluent_indx)
            print("..compute_effect_value: ", fluent_name, " - indx: ", fluent_indx)
        
            if DEBUG>4:
                print("Effect:", effect_data.effect, " fl_name: ", fluent_name)

        are_prec_satisfied=1.0 #self._are_preconditions_satisfied
        sympy_function=effect_data.sympy_sat_function        

        if (are_prec_satisfied < 0.0):
            sympy_function=effect_data.sympy_unsat_function
        
        result= sympy_function(predicates_indexes, state_values)

        if DEBUG>2:
            value=float(state_values[fluent_name])
            print("..compute_effect_value: ", fluent_name, " - indx: ", effect_indx)
            print("-->",fluent_name,"- init:",value, " delta:", result-value, " new: ", result)

            if(DEBUG>5):
                print("Result: ", result)
                tf.print("TResult: ", result)
        
        return (fluent_indx, result)

    
    def _apply_single_effect(self, effect_indx, predicates_indexes, state_values):
        """
        Apply a single effect to the new state.

        Args:
            effect (Effect): The effect to apply.
            curr_state (dict): The initial state.
        """
        if DEBUG>5: #XXX
            print("...Apply single effect: ", effect_indx, " predicates: ", predicates_indexes)
        effect_data = GlobalData._class_effects_list[effect_indx]
        #fl_name=GlobalData._class_predicates_list_string[ predicates_indexes[effect_data.effect_predicates_position]]
        fluent_indx=predicates_indexes[effect_data.effect_predicates_position]
        #fluent_name = GlobalData.get_key_from_indx_predicates_list( fluent_indx)

        if DEBUG>4:
            print("Effect:", effect_data.effect, " fl_name: ", fluent_name)
        result=0.0

        if(DEBUG>2):
            print("..compute_effect_value: ", fluent_name, " - indx: ", fluent_indx)
        
        are_prec_satisfied=1.0 #self._are_preconditions_satisfied
        sympy_expr = effect_data.sympy_sat
        result=0.0
        if (are_prec_satisfied < 0.0):
            sympy_expr = effect_data.sympy_unsat
        
        #result= SympyToTfConverter.convert_new(sympy_expr, predicates_indexes , state, are_prec_satisfied)
        result=self.converter.convert(sympy_expr, predicates_indexes , are_prec_satisfied, state_values)
        
        if(DEBUG>5):
            print("Result: ", result)
            tf.print("TResult: ", result)

        if DEBUG>2:
            value=float(state_values[fluent_name])
            print("..compute_effect_value: ", fluent_name, " - indx: ", effect_indx)
            print("-->",fluent_name,"- init:",value, " delta:", result-value, " new: ", result)

        return (fluent_indx, result)

    @abstractmethod
    def evaluate_preconditions(self, predicates_indexes, state_values):
        """
        Evaluates all preconditions and returns whether they are satisfied.
        
        Args:
            state (dict): The current state of the problem (variables).
            
        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """
        pass



    @abstractmethod
    def apply_effects(self, predicates_indexes, state_values):
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
                f"action={self._action}, "
                f"is applicable={self._are_preconditions_satisfied}, "
                #f"state={self.new_state}) "
                ) 
    


class TfLiftedAction (TensorAction):
    _class_action_id = 0  # Class attribute to keep track of the class ID

    def __init__(self, problem, plan_action, converter, tensor_state):

        super().__init__(problem, plan_action, converter, tensor_state)
        self._lifted_action=plan_action._action
        
        if self._lifted_action in GlobalData._class_liftedData_map:
            lifted_indx=GlobalData._class_liftedData_map[self._lifted_action]
            liftedData = GlobalData._class_liftedData_list[lifted_indx]
            self._lifted_action=liftedData.lifted_action
            self._lifted_action_indx=lifted_indx
            self._predicates_list=liftedData.predicates_list
            self._predicates_indexes=liftedData.predicates_indexes
            self._act_preconditions_list=liftedData.act_preconditions_list
            self._act_preconditions_function=liftedData.act_preconditions_function
            self._act_effects_lifted_list=liftedData.act_effects_list
            self._act_effects_lifted_pos_list=liftedData.act_effect_pos_list
            self.apply_action_funct=liftedData.apply_action_funct 
            self.apply_action_concrete_funct=liftedData.apply_action_concrete_funct
   
        else:
            if DEBUG>2:
                print("Create TfLiftedAction:", self._action.name, " id:", self._action_id)
            
            fve = up.model.walkers.FreeVarsExtractor()
            predicates_set= set()
            for prec in plan_action.action.preconditions:
                predicates_set.update(fve.get(prec))
            for effect in plan_action.action.effects:
                predicates_set.update(fve.get(effect.fluent))
                predicates_set.update(fve.get(effect.value))
                if effect.is_conditional():
                    predicates_set.update(fve.get(effect.condition))
            self._predicates_list=list(predicates_set) #Lifted predicate set
            self._predicates_indexes=GlobalData.insert_predicates_in_map(predicates_set)
            self.tensor_state.insert_zero(predicates_set)
                        
            self._build_preconditions_effects()

            self._lifted_action_indx=len(GlobalData._class_liftedData_list)

            self.apply_action_funct=TfLiftedAction.apply_TfLiftedAction_function(self._lifted_action_indx)
            
            effect_pos_list=[]
            funct_list=[]
            for ef_indx in self._act_effects_lifted_list:
                effect_data=GlobalData._class_effects_list[ef_indx]
                pos=effect_data.effect_predicates_position
                funct=GlobalData._class_effects_list[ef_indx].sympy_sat_function
                effect_pos_list.append(pos)
                funct_list.append(funct)
                if DEBUG>5:
                    print("Effect:", effect_data.effect, " fl_name: ", effect_data.name)
            
            self._act_effects_lifted_pos_list=effect_pos_list
            liftedData=LiftedActionData(self._lifted_action, 
                                        self._lifted_action_indx,
                                        self._predicates_list, 
                                        self._predicates_indexes, 
                                        self._act_preconditions_list, 
                                        self._act_preconditions_function, 
                                        self._act_effects_lifted_list, 
                                        self._act_effects_lifted_pos_list, 
                                        self.apply_action_funct)
            GlobalData._class_liftedData_list.append(liftedData)
            GlobalData._class_liftedData_map[self._lifted_action] = self._lifted_action_indx
            GlobalData._class_lifted_effects_pos_list.append(tf.constant(effect_pos_list, dtype=tf.int32))
            GlobalData._class_lifted_effects_funct_list.append(funct_list) 
            GlobalData._class_act_preconditions_function_list.append(self._act_preconditions_function)
            #branch_list =[
            #   (lambda f=f: f(predicates_indexes, state_values))
            #    for f in funct_list 
            #]
            concrete_funct=self.apply_action_funct.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.int32), tf.TensorSpec(shape=[None], dtype=tf.float32))
            liftedData.set_concrete_funct(concrete_funct)
            self.apply_action_concrete_funct=concrete_funct #XXXX
            
            

    def _build_preconditions_effects(self):
        """
        Apply the effects of the action to the new state.

        Args:
            effects (list): A list of effect objects to apply.
            new_state (dict): The new state that will be modified.
        """
        #PRECONDITIONS
        preconditions = self._action.preconditions
        for prec in preconditions:
            if DEBUG>5:
                print("Define precondition:", prec)
            cond_position = self.store_condition(prec)
            # Store the precondition position to build the action's preconditions_list
            self._act_preconditions_list.append(cond_position)
            
        if len(self._act_preconditions_list)>1:
            self._act_preconditions_function=lambda p,s: tf.add_n([ GlobalData._class_conditions_list[f].sympy_function(p,s) for f in self._act_preconditions_list])
        else:
            self._act_preconditions_function=GlobalData._class_conditions_list[self._act_preconditions_list[0]].sympy_function  

        #EFFECTS
        effects = self._action.effects
        for effect in effects:
            ef_position= self.store_effect(effect)
            # Store the effect position to build the action's effects_list
            self._act_effects_lifted_list.append(ef_position)


    def get_lifted_action_indx(self):
        return self._lifted_action_indx

    def store_condition(self, prec): # prec: up.model.Precondition): CHECK
        """
        Store a precondition in the class list and map if it does not already exist.
        """
        converter= self.converter

        return TensorAction.store_condition(prec, converter, self._predicates_list)
    


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
            fl_name = effect.fluent.get_name()  # Assuming fluent() provides the name
            ef_str= str(effect.value)

            effect_set= self._predicates_list
            effect_indexes= self._predicates_indexes
   
            cond_position=-1 #No conditional effect
            if effect.is_conditional():
                cond_position=self.store_condition(effect.condition)
  
            # Compute sympy expressions
            effect_fluent_predicates_position=effect_set.index(effect.fluent)
            sympy_expr_sat = converter.when_sympy_expr_not_inserted(effect, effect_set, cond_position, 1.0)
            sympy_expr_unsat = converter.when_sympy_expr_not_inserted(effect, effect_set, cond_position, -1.0)
              
            sympy_sat_function=SympyToTfConverter.sympy_to_tensor_function(sympy_expr_sat)
            sympy_unsat_function=SympyToTfConverter.sympy_to_tensor_function(sympy_expr_unsat)
            ef_position = len(GlobalData._class_effects_list)
            # Create an EffectData object
            effect_data = EffectData(
                effect=effect,
                position=ef_position,
                effect_predicates_position=effect_fluent_predicates_position,
                name=fl_name,
                condition=cond_position,
                sympy_sat=sympy_expr_sat,
                sympy_sat_function=sympy_sat_function,
                sympy_unsat=sympy_expr_unsat,
                sympy_unsat_function=sympy_unsat_function,
                predicates_keys=effect_set,
                predicates_indexes=effect_indexes
            )

            # Store in list and map
            GlobalData._class_effects_list.append(effect_data)
            GlobalData._class_effects_map[effect] = ef_position
            if DEBUG > 5:
                print(f"Added new effect: {effect_data}")
            
        return ef_position


    #@tf.function#(,input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32), tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def apply_TfLiftedAction_old(act_index:int, predicates_indexes:tf.TensorSpec(shape=(), dtype=tf.int32), state_values:tf.TensorSpec(shape=(), dtype=tf.float32)): #: up.tensor.TensorState):
        """
        Apply an action to the state if the preconditions are met, and manage the effects.
        
        Args:
            curr_state (dict): The initial state as a dictionary of fluent names to values.
        
        Returns:
            cost (Tensor): The cost calculated by applying the action's effects, or an invalid state if preconditions are not met.
        """
        print("Apply lifted action: ", act_index)
        # Evaluate preconditions
        liftedData = GlobalData._class_liftedData_list[act_index]
        are_preconditions_satisfied = liftedData.act_preconditions_function(predicates_indexes, state_values)
        #indexes=tf.Variable(dtype=tf.int32, size=0, dynamic_size=True)
        #values=tf.Variable(dtype=tf.float32, size=0, dynamic_size=True)
        indexes=list()
        values=list()
        # Apply effects if preconditions are satisfied
        #state_update=self.apply_effects(predicates_indexes, state_values)
        for effect_indx in liftedData.act_effects_list:
            effect_data=GlobalData._class_effects_list[effect_indx]
            if DEBUG>5:
                print("Applying effect:", effect_data.effect)
            result=TF_SAT
                          
            pos=predicates_indexes[effect_data.effect_predicates_position]
            value=GlobalData._class_effects_list[effect_indx].sympy_sat_function(predicates_indexes, state_values) #executed here for avoid tf.function out of scope error 
            if tf.greater_equal(effect_data.condition,TF_INT_ZERO):    
                result= GlobalData._class_conditions_list[effect_data.condition].sympy_function(predicates_indexes,state_values)
                #if DEBUG>4:
                #    print("Condition result:", result)
                #Needed in this position (not after if) for avoid tf.function out of scope error 
                if tf.less(result,TF_ZERO):
                    value=TF_ZERO
            
            if tf.greater_equal(result,TF_ZERO) and tf.not_equal(value,TF_ZERO):
                indexes.append(pos)
                values.append(value)
        


        metric_value_add = UNSAT_PENALTY + UNSAT_PENALTY * (-1.0) * tanh(are_preconditions_satisfied)
        metric_value_add = tf.cond(tf.math.greater_equal(are_preconditions_satisfied, TF_ZERO),lambda: TF_ZERO,lambda: metric_value_add)
        if tf.math.less(are_preconditions_satisfied, TF_ZERO):
            indexes.append(GlobalData.pos_metric_expr)
            values.append(metric_value_add)
            #tf.print("Metric pos: ", GlobalData.pos_metric_expr, " value: ", metric_value_add, " aps: ", are_preconditions_satisfied)
            

        if DEBUG>0:
            if are_preconditions_satisfied<0:    
                print("Preconditions not satisfied, action id: ", liftedData._lifted_action_indx)
            else:
                print("Preconditions satisfied, action id: ", liftedData._lifted_action_indx)
            print()

        indexes=tf.reshape(tf.stack(indexes), (-1, 1))
        values=tf.stack(values)
        #tf.print((indexes),": ",(values)," len equal: ",len(indexes)==len(values)," \n")

        if tf.shape(indexes)[0] > 0:  # Apply updates in batch
            state_values.scatter_nd_add(indices=indexes, updates=values)


        return are_preconditions_satisfied


    
    def apply_TfLiftedAction(
        act_index: int,
        predicates_indexes: tf.TensorSpec(shape=(), dtype=tf.int32),
        state_values: tf.TensorSpec(shape=(), dtype=tf.float32),
    ):
        """
        Apply an action to the state if the preconditions are met, and manage the effects.
        
        Args:
            act_index (int): The index of the action.
            predicates_indexes (TensorSpec): Tensor of predicate indexes.
            state_values (TensorSpec): Tensor of state values.
        
        Returns:
            are_preconditions_satisfied (Tensor): Whether preconditions are met.
            return_values (Tensor): The calculated effect values.
        """

        print("Apply lifted action:", act_index)
        lifted_effects_funct = GlobalData._class_lifted_effects_funct_list[act_index]
        act_preconditions_function = GlobalData._class_act_preconditions_function_list[act_index]

        # Evaluate preconditions
        are_preconditions_satisfied = act_preconditions_function(predicates_indexes, state_values)
        
        # Compute effects using list comprehension
        values_list = [effect_funct(predicates_indexes, state_values) for effect_funct in lifted_effects_funct]
        
        # Compute metric value update
        metric_value_add = UNSAT_PENALTY + UNSAT_PENALTY * (-1.0) * tf.tanh(are_preconditions_satisfied)
        satisfied = are_preconditions_satisfied >= TF_ZERO
        metric_value_add = tf.cond(satisfied, lambda: TF_ZERO, lambda: metric_value_add)
        
        # Append metric value update and satisfaction flag
        values_list.append(metric_value_add)
        values_list.append(tf.cast(satisfied, dtype=tf.float32))
        
        return_values = tf.stack(values_list)
        return are_preconditions_satisfied, return_values



    #@tf.function
    def apply_TfLiftedAction_for(act_index:int, predicates_indexes:tf.TensorSpec(shape=(), dtype=tf.int32), state_values:tf.TensorSpec(shape=(), dtype=tf.float32)): #: up.tensor.TensorState):
        """
        Apply an action to the state if the preconditions are met, and manage the effects.
        
        Args:
            curr_state (dict): The initial state as a dictionary of fluent names to values.
        
        Returns:
            cost (Tensor): The cost calculated by applying the action's effects, or an invalid state if preconditions are not met.
        """

        print("Apply lifted action: ", act_index)
        
        #liftedData = GlobalData._class_liftedData_list[act_index]  
        #lifted_effects_pos=GlobalData._class_lifted_effects_pos_list[act_index]
        lifted_effects_funct=GlobalData._class_lifted_effects_funct_list[act_index]
        act_preconditions_function=GlobalData._class_act_preconditions_function_list[act_index]
        
        #effects_pos=tf.gather(predicates_indexes,lifted_effects_pos)
        
        max_size=len(lifted_effects_funct)+2 # metric expression, and unsat precondtions
        #indexes = tf.TensorArray(dtype=tf.int32, size=max_size, dynamic_size=False)
        values = tf.TensorArray(dtype=tf.float32, size=max_size, dynamic_size=False)  
        
        # Evaluate preconditions
        are_preconditions_satisfied = act_preconditions_function(predicates_indexes, state_values)
        
        step=0
        for effect_funct in lifted_effects_funct:
     
            #pos = effects_pos[step]
            value = effect_funct(predicates_indexes, state_values)
                  
            #indexes=indexes.write(step, pos) # tf.cond(cond, lambda: pos, lambda: -1))  # -1 as placeholder
            values=values.write(step, value) #tf.cond(cond, lambda: value, lambda: TF_ZERO))   
            step+=1   

        # Compute metric value update
        metric_value_add = UNSAT_PENALTY + UNSAT_PENALTY * (-1.0) * tf.tanh(are_preconditions_satisfied)
        satisfied=(are_preconditions_satisfied >= TF_ZERO) # tf.greater_equal(are_preconditions_satisfied, TF_ZERO)
        metric_value_add = tf.cond(satisfied, lambda: TF_ZERO, lambda: metric_value_add)
    

        #indexes=indexes.write(step, GlobalData.pos_metric_expr)
        values=values.write(step, metric_value_add)  
        step+=1
        values=values.write(step, tf.cast(satisfied, dtype=tf.float32))   
        
        #if DEBUG>0:
        #    if are_preconditions_satisfied<0:    
        #        print("Preconditions not satisfied, action id: ", liftedData._lifted_action_indx)
        #    else:
        #        print("Preconditions satisfied, action id: ", liftedData._lifted_action_indx)
        #    print()

        #return_indexes=tf.reshape(indexes.stack(), (-1, 1))
        return_values=values.stack()
    
        #tf.print("UpdateInside2: ",(indexes),": ",(values)," \n")
        return are_preconditions_satisfied,return_values

    #@tf.function
    def apply_TfLiftedAction_liftedData(act_index:int, predicates_indexes:tf.TensorSpec(shape=(), dtype=tf.int32), state_values:tf.TensorSpec(shape=(), dtype=tf.float32)): #: up.tensor.TensorState):
        """
        Apply an action to the state if the preconditions are met, and manage the effects.
        
        Args:
            curr_state (dict): The initial state as a dictionary of fluent names to values.
        
        Returns:
            cost (Tensor): The cost calculated by applying the action's effects, or an invalid state if preconditions are not met.
        """

        print("Apply lifted action: ", act_index)
        
        liftedData = GlobalData._class_liftedData_list[act_index]  
        #lifted_effects_pos=GlobalData._class_lifted_effects_pos_list[act_index]
        #lifted_effects_funct=GlobalData._class_lifted_effects_funct_list[act_index]
        #act_preconditions_function=GlobalData._class_act_preconditions_function_list[act_index]
        
        #effects_pos=tf.gather(predicates_indexes,lifted_effects_pos)
        
        max_size=len(liftedData.act_effects_list)+1
        indexes = tf.TensorArray(dtype=tf.int32, size=max_size, dynamic_size=False)
        values = tf.TensorArray(dtype=tf.float32, size=max_size, dynamic_size=False)  
        
        # Evaluate preconditions
        are_preconditions_satisfied = liftedData.act_preconditions_function(predicates_indexes, state_values)

        
        step=0        
        for effect_indx in liftedData.act_effects_list:
            effect_data=GlobalData._class_effects_list[effect_indx]
       
            pos = predicates_indexes[effect_data.effect_predicates_position]
            value = effect_data.sympy_sat_function(predicates_indexes, state_values)
          
            indexes=indexes.write(step, pos) # tf.cond(cond, lambda: pos, lambda: -1))  # -1 as placeholder
            values=values.write(step, value) #tf.cond(cond, lambda: value, lambda: TF_ZERO))   
            step+=1   

        # Compute metric value update
        metric_value_add = UNSAT_PENALTY + UNSAT_PENALTY * (-1.0) * tf.tanh(are_preconditions_satisfied)
        metric_value_add = tf.cond(are_preconditions_satisfied >= TF_ZERO, lambda: TF_ZERO, lambda: metric_value_add)

        indexes=indexes.write(step, GlobalData.pos_metric_expr)
        values=values.write(step, metric_value_add)  
        
        #if DEBUG>0:
        #    if are_preconditions_satisfied<0:    
        #        print("Preconditions not satisfied, action id: ", liftedData._lifted_action_indx)
        #    else:
        #        print("Preconditions satisfied, action id: ", liftedData._lifted_action_indx)
        #    print()

        return_indexes=tf.reshape(indexes.stack(), (-1, 1))
        return_values=values.stack()
    
        #tf.print("UpdateInside2: ",(indexes),": ",(values)," \n")
        return are_preconditions_satisfied,return_indexes,return_values

    
    # Wrap the conversion into a tf.function
    def apply_TfLiftedAction_function(act_indx):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32), tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def apply_funct(predicates_indexes, state_values):
            return TfLiftedAction.apply_TfLiftedAction( act_indx, predicates_indexes,state_values)
            
        return apply_funct
        

    #@tf.function
    def evaluate_preconditions(self, predicates_indexes, state_values):
        """
        Evaluates all preconditions and returns whether they are satisfied.
        
        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """
        
        if DEBUG>5:
            print("Evaluate preconditions, act: ", self.get_name(), " id:", self._action_id)

            
        preconditions = self._act_preconditions_list
        satisfied = tf.constant(0.0)  # Assume preconditions are satisfied initially; negative value indicates unsatisfied

        for prec in preconditions:
            if DEBUG>2:            
                print("Evaluating precondition:", prec)
            #value = self._evaluate_condition(prec, predicates_indexes, state_values)
            value = TensorAction.evaluate_condition(prec, predicates_indexes, state_values)
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
    
    def apply_effects(self, predicates_indexes, state_values): #up_id useul for tf.function retracing
        """
        Apply the effects of the action to the new state.

        Args:
            effects (list): A list of effect objects to apply.
            new_state (dict): The new state that will be modified.
        """
       
        if DEBUG>5: #XXX
            print("\nApply effects, act: ", self.get_name(), " id:", self.get_action_id()," predicates: ", predicates_indexes)
        effects = self._act_effects_lifted_list
        out_result=list()
        for effect_indx in effects:
            effect_data=GlobalData._class_effects_list[effect_indx]
            if DEBUG>5:
                print("Applying effect:", effect_data.effect)
            result=1.0
            if effect_data.condition>=0:
                #result = self._evaluate_condition(effect_data.condition, predicates_indexes, state_values) 
                result=TensorAction.evaluate_condition(effect_data.condition, predicates_indexes, state_values)
                if DEBUG>4:
                    tf.print("Condition: ", effect_data.condition ," result:", result)
            
            #needed in this position (not after if) for avoid tf.function out of scope error
            pos,value=self._apply_single_effect(effect_indx, predicates_indexes, state_values)  
            #pos,value=predicates_indexes[effect_data.effect_predicates_position],GlobalData._class_effects_list[effect_indx].sympy_sat_function(predicates_indexes, state_values)
            if tf.less(result, 0):
                pos=-1
                #tf.print("Effect: ", effect_data.effect, " pos: ", pos, " value: ", value, " condition: ", result)
            else:
                #tf.print("Effect executed: ", effect_data.effect, " pos: ", pos, " value: ", value, " condition: ", result)
                out_result.append((pos,value))
        
        return out_result

    #@tf.function
    def apply_single_effect(self, effect_indx, predicates_indexes):
        return TensorAction.apply_single_effect(effect_indx, predicates_indexes, self.tensor_state.get_state_values())


    
    def __repr__(self):
        return (f"TfLiftedAction(action_id={self._action_id}, "
                f"lifted_action={self._action}, preconditions={self._action.preconditions}, "
                f"effects={self._action.effects})")


class TfAction(TfLiftedAction):

    def __init__(self, problem: up.model.Problem, plan_action: up.plans.plan.ActionInstance, converter, tensor_state):
        """
        Initialize the TensorAction with the problem, action, and converter.
        
        Args:
            problem (Problem): The problem containing the objective and other parameters.
            action (Action): The action to apply.
        """
        super().__init__(problem, plan_action, converter, tensor_state)
        #pk = problem.kind
        #if not Grounder.supports(pk):
        #    msg = f"The Grounder used in the {type(self).__name__} does not support the given problem"
        #    if self.error_on_failed_checks:
        #        raise UPUsageError(msg)
        #    else:
        #        warn(msg)
        self._grounder = GrounderHelper(problem,prune_actions=False)
        grounded_action = self._grounder.ground_action(plan_action._action, plan_action.actual_parameters )
        if grounded_action is None:
            raise UPInvalidActionError("Apply_unsafe got an inapplicable action.")
        self._action = grounded_action # Store the action
 
        if DEBUG>1:
            print("TfAction ID:", self._action_id)
        
        # Create a lifted action if not already created
        self._lifted_act_subs: Dict[up.model.Expression, up.model.Expression] = dict(
                    zip(plan_action.action.parameters, list(plan_action.actual_parameters))
                )
        lifted_predicates_set = self.get_predicates_list()
        predicates_list = []
        predicates_id_list=[]
        pred_matchings = dict()
        pred_matchings_lifted_data = dict()
        keys=list()
        for pred in lifted_predicates_set:
            if pred.node_type != OperatorKind.FLUENT_EXP :
                print("Error: Not a fluent expression: ", pred)
                continue
            new_pred = pred.substitute(self._lifted_act_subs)
            pred_matchings.update({str(pred): str(new_pred)})
            pred_matchings_lifted_data.update({pred: new_pred})
            predicates_list.append(new_pred)
            indx=GlobalData._class_predicates_map.lookup(new_pred.get_name())
            if (indx<0):
                pred_name=new_pred.get_name()
                key=[pred_name]
                keys.append(pred_name)
                indexes=GlobalData.insert_predicates_in_map(key) #returns a list of indexes since I provided a list
                indx=indexes[0]
            predicates_id_list.append(indx)

        self.tensor_state.insert_zero(keys) 
        self._pred_matchings_lifted_data = pred_matchings_lifted_data
        self._pred_matchings = pred_matchings
        self._up_act_free_vars = lifted_predicates_set

        self._predicates_list = predicates_list   
        self._predicates_indexes = tf.constant([t.numpy() for t in predicates_id_list], dtype=tf.int32)
        self._state_predicates_indexes=tf.constant([self.tensor_state.get_key_position(str(pred)) for pred in predicates_list])
    
        self._act_effects_pos_list=tf.reshape(tf.concat([
            tf.gather(self._state_predicates_indexes, self._act_effects_lifted_pos_list),
            tf.expand_dims(GlobalData.pos_metric_expr, axis=0),
            tf.expand_dims(GlobalData.pos_are_prec_sat, axis=0)], axis=0), (-1, 1))#Add also the metric position in order to consider penalization with unsatisfied preconditions
    
        effects = self._action.effects
        self.act_name_effects_list=[]
        for effect in effects:
            ef_name=effect.fluent.get_name()
            self.act_name_effects_list.append(ef_name)

    def get_predicates_indexes(self):
        return self._state_predicates_indexes

    def get_effects_pos(self):
        return self._act_effects_pos_list

    def apply_action(self, state_values): #: up.tensor.TensorState):    
        """
        Apply TensorfFlow  action to the state if the preconditions are met, and manage the effects.
        """
        update_state= super().apply_action( self._state_predicates_indexes, state_values)        
        #self.set_are_preconditions_satisfied(self._tf_lifted_action.get_are_preconditions_satisfied())
        return update_state    


    def set_curr_state(self, state):
        self.curr_state=state
        self._tf_lifted_action.set_curr_state(state)
        
