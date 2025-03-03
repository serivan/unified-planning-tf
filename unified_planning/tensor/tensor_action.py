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
from unified_planning.tensor.constants import *
from unified_planning.engines.compilers import Grounder, GrounderHelper
from unified_planning.tensor.tensor_state import TensorState, TfState
from unified_planning.tensor.tensor_fluent import TensorFluent, TfFluent

from unified_planning.tensor.converter import SympyToTfConverter

from tensorflow.lookup.experimental import MutableHashTable

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
    def __init__(self, effect, position, effect_predicates_position, name, condition, sympy_sat, sympy_unsat, predicates_keys, predicates_indexes):
        self.effect = effect
        self.position = position
        self.effect_predicates_position=effect_predicates_position
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


# TensorAction class modified to use the SympyToTfConverter
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
        self.curr_state=state
        self.new_state=None
        self._plan_action = plan_action
        self._action_id = TensorAction._class_action_id  # Assign a unique ID to the action
        TensorAction._class_action_id += 1  # Increment the class ID for the next action
        self._act_effects_list=[]
        self._act_preconditions_list=[]
        self._predicates_list=None
        self._predicates_indexes=None

        self._are_preconditions_satisfied=0
        self.converter=converter
        if state is not None:
            self.curr_state = state
            self.converter.set_state(self.curr_state)


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
        return self._action.name
    
    def get_predicates_list(self):
        return self._predicates_list
    
    #@tf.function
    def apply_action(self, predicates_indexes, curr_state: MutableHashTable): #: up.tensor.TensorState):
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
            #TensorState.print_filtered_hash_state(self.curr_state,["next"])
            #print("Current state:", self.curr_state)
            if tf.not_equal(self.curr_state.lookup("objective"), MISSING_VALUE):
                tf.print(".Current state objective:", self.curr_state['objective'])

        # Evaluate preconditions
        self._are_preconditions_satisfied = self.evaluate_preconditions( predicates_indexes, self.curr_state)
        if DEBUG>1:
            if self._are_preconditions_satisfied<0:    
                print("Preconditions not satisfied, action id: ", self._action_id, " name: ", self.get_name())
        # Apply effects if preconditions are satisfied
        state_update=self.apply_effects(predicates_indexes)

        state_update.append((ARE_PREC_SATISF_STR,self._are_preconditions_satisfied)) #UPDATE state_update to export value for tf.function
   
 
        #   if pos<0:
        #        pos=len(state_update)
     
        #if pos==len(state_update):
        #    state_update.append((metric_expr, metric_value))
        #elif pos>=0:
        #    del state_update[pos]
        #    state_update.append((metric_expr, metric_value))

        #state_update[str(metric.expression)]=metric_value

        if DEBUG>0:
            if self._are_preconditions_satisfied<0:    
                print("Preconditions not satisfied, action id: ", self._action_id, " name: ", self.get_name())
            else:
                print("Preconditions satisfied, action id: ", self._action_id, " name: ", self.get_name())
            print("Metric value: ", metric_value)
            print()

        if DEBUG>1:
            print("Update after action:", end=":: ")
            #TensorState.print_filtered_dict_state(state_update,["next"])
            print(state_update)
        return state_update

  

    def store_condition(prec, converter,  predicates_list=None): # prec: up.model.Precondition): CHECK
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
            predicates_indexes=GlobalData.insert_predicates_in_map(predicates_set)
                        
            if predicates_list is None:                 
                predicates_list=list(predicates_set)
            else:
                indexes=GlobalData.get_values_from_predicates_list(predicates_list)
                predicates_indexes=tf.constant([t.numpy() for t in indexes], dtype=tf.int32)

            lifted_prec_str=GlobalData.get_lifted_string(prec_str,predicates_list)
            sympy_expr =  converter.define_condition_expr(lifted_prec_str) 

            cond_position = len(GlobalData._class_conditions_list)
            # Create a PreconditionData object
            prec_data = PreconditionData(
                prec=prec,
                position=cond_position,
                name=fl_name,
                sympy_expr=sympy_expr,
                predicates_keys=predicates_list,
                predicates_indexes=predicates_indexes
            )
            # Store in list and map
            GlobalData._class_conditions_list.insert(cond_position, prec_data)
            GlobalData._class_conditions_map[prec] = cond_position
            if DEBUG > 5:
                print(f"Added new precondition: {prec_data}")
        return cond_position
    

    def evaluate_condition(condition: int, state, predicates_indexes=None):
        """
        Evaluates a single precondition and returns the resulting value.
        
        Args:
            precondition (Precondition): A precondition to evaluate.
        
        Returns:
            Tensor: The evaluation result of the precondition.
        """
        sympy_expr = GlobalData._class_conditions_list[condition].sympy_expr
        if predicates_indexes is None:
            predicates_indexes=GlobalData._class_conditions_list[condition].predicates_indexes
        value=  SympyToTfConverter.convert_new(sympy_expr, predicates_indexes, state)
        
        return value 


    def _evaluate_condition(self, condition: int, state, predicates_indexes=None):
        """
        Evaluates a single precondition and returns the resulting value.
        
        Args:
            precondition (Precondition): A precondition to evaluate.
        
        Returns:
            Tensor: The evaluation result of the precondition.
        """
        sympy_expr = GlobalData._class_conditions_list[condition].sympy_expr
        if predicates_indexes is None:
            predicates_indexes=GlobalData._class_conditions_list[condition].predicates_indexes
        value=  self.converter.convert(sympy_expr, predicates_indexes, state)
        
        return value 

    def get_next_state(self):
        return self.new_state
    def get_are_preconditions_satisfied (self):
        return self._are_preconditions_satisfied
    
    def set_are_preconditions_satisfied (self, value):
        self._are_preconditions_satisfied=value

    def is_applicable (self):
        return self._are_preconditions_satisfied>=0
    
    #
    #@tf.function
    def _apply_single_effect(self, effect_indx, predicates_indexes, state):
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
        fl_name = GlobalData.get_key_from_indx_predicates_list( predicates_indexes[effect_data.effect_predicates_position])
        #fl_name=fl_string.numpy().decode("utf-8")

        if DEBUG>4:
            print("Effect:", effect_data.effect, " fl_name: ", fl_name)
        result=0.0

        if(DEBUG>2):
            print("..compute_effect_value: ", fl_name, " - indx: ", effect_indx)
        
        are_prec_satisfied=1.0 #self._are_preconditions_satisfied
        sympy_expr = effect_data.sympy_sat
        result=0.0
        if (are_prec_satisfied < 0.0):
            sympy_expr = effect_data.sympy_unsat
        
        if predicates_indexes is None:
            predicates_indexes=effect_data.predicates_indexes

        result= SympyToTfConverter.convert_new(sympy_expr, predicates_indexes , state, are_prec_satisfied)
        #result= self.converter.convert(sympy_expr, predicates_indexes , are_prec_satisfied)
        if(DEBUG>5):
            print("Result: ", result)
            tf.print("TResult: ", result)

        if DEBUG>2:
            value=float(state[fl_name])
            print("..compute_effect_value: ", fl_name, " - indx: ", effect_indx)
            print("-->",fl_name,"- init:",value, " delta:", result-value, " new: ", result)

        return (fl_name, result)

    #
    #@tf.function
    def apply_single_effect(effect_indx, predicates_indexes, state):
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
        fl_name = GlobalData.get_key_from_indx_predicates_list( predicates_indexes[effect_data.effect_predicates_position])
        #fl_name=fl_string.numpy().decode("utf-8")

        if DEBUG>4:
            print("Effect:", effect_data.effect, " fl_name: ", fl_name)
        result=0.0

        if(DEBUG>2):
            print("..compute_effect_value: ", fl_name, " - indx: ", effect_indx)
        
        are_prec_satisfied=1.0 #self._are_preconditions_satisfied
        sympy_expr = effect_data.sympy_sat
        result=0.0
        if (are_prec_satisfied < 0.0):
            sympy_expr = effect_data.sympy_unsat
        
        if predicates_indexes is None:
            predicates_indexes=effect_data.predicates_indexes

        result= SympyToTfConverter.convert_new(sympy_expr, predicates_indexes , state, are_prec_satisfied)
        
        if(DEBUG>5):
            print("Result: ", result)
            tf.print("TResult: ", result)

        if DEBUG>2:
            value=float(state[fl_name])
            print("..compute_effect_value: ", fl_name, " - indx: ", effect_indx)
            print("-->",fl_name,"- init:",value, " delta:", result-value, " new: ", result)

        return (fl_name, result)

    @abstractmethod
    def evaluate_preconditions(self, predicates_indexes=None, state=None):
        """
        Evaluates all preconditions and returns whether they are satisfied.
        
        Args:
            state (dict): The current state of the problem (variables).
            
        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """
        pass



    @abstractmethod
    def apply_effects(self, predicates_indexes=None):
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

    def __init__(self, problem, plan_action, converter, state):

        super().__init__(problem, plan_action, converter, state)
        self._action=self._lifted_action=plan_action._action
        
        if self._lifted_action in GlobalData._class_lifted_actions_id_map:
            self._lifted_action_id = GlobalData._class_lifted_actions_id_map[self._lifted_action]
        else:
            self._lifted_action_id=len(GlobalData._class_lifted_actions_id_map)
            GlobalData._class_lifted_actions_id_map[self._lifted_action]=self._lifted_action_id


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
                        
        self._build_preconditions_effects()


    def get_lifted_action_id(self):
        return self._lifted_action_id


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
            ef_position = len(GlobalData._class_effects_list)
            fl_name = effect.fluent.get_name()  # Assuming fluent() provides the name
            ef_str= str(effect.value)

            #fve = up.model.walkers.FreeVarsExtractor()
            #effect_set= fve.get(effect.fluent)| fve.get(effect.value) |fve.get(effect.condition)
            #effect_indexes=GlobalData.insert_predicates_in_map(effect_set)

            effect_set= self._predicates_list
            effect_indexes= self._predicates_indexes

            # Compute sympy expressions
            effect_fluent_predicates_position=effect_set.index(effect.fluent)
            sympy_expr_sat = converter.when_sympy_expr_not_inserted(effect, effect_set, 1.0)
            sympy_expr_unsat = converter.when_sympy_expr_not_inserted(effect, effect_indexes, -1.0)
            cond_position=-1 #No conditional effect
            if effect.is_conditional():
                cond_position=self.store_condition(effect.condition)
                

            # Create an EffectData object
            effect_data = EffectData(
                effect=effect,
                position=ef_position,
                effect_predicates_position=effect_fluent_predicates_position,
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
            

        #EFFECTS
        effects = self._action.effects
        for effect in effects:
            ef_position= self.store_effect(effect)
            # Store the effect position to build the action's effects_list
            self._act_effects_list.append(ef_position)


    def __repr__(self):
        return (f"TfLiftedAction(action_id={self._action_id}, "
                f"lifted_action={self._action}, preconditions={self.preconditions}, "
                f"effects={self.effects})")



    def evaluate_condition(self, condition: int, predicates_indexes):
        """
        Evaluates a single precondition and returns the resulting value.
        
        Args:
            precondition (Precondition): A precondition to evaluate.
        
        Returns:
            Tensor: The evaluation result of the precondition.
        """
        converter= self.converter
        return self._evaluate_condition(condition, self.curr_state, predicates_indexes)

    @tf.function
    def evaluate_preconditions(self, predicates_indexes,  state=None):
        """
        Evaluates all preconditions and returns whether they are satisfied.
        
        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """
        if DEBUG>5:
            print("Evaluate preconditions, act: ", self.get_name(), " id:", self._action_id)

        if state is not None:
            self.curr_state = state
            self.converter.set_state(state)
            
        preconditions = self._act_preconditions_list
        satisfied = tf.constant(0.0)  # Assume preconditions are satisfied initially; negative value indicates unsatisfied

        for prec in preconditions:
            if DEBUG>2:            
                print("Evaluating precondition:", prec)
            #value = self.evaluate_condition(prec, predicates_indexes)
            value = TensorAction.evaluate_condition(prec, self.curr_state, predicates_indexes)
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

    @tf.function
    def apply_effects(self, predicates_indexes): #up_id useul for tf.function retracing
        """
        Apply the effects of the action to the new state.

        Args:
            effects (list): A list of effect objects to apply.
            new_state (dict): The new state that will be modified.
        """
        # Update the metric value if the preconditions are not 
        #metric=self.problem.quality_metrics[0]
        #metric_expr=tf.constant(str(metric.expression), tf.string)
        #metric_value_found=False
        #metric_value=0.0        

        if DEBUG>5: #XXX
            print("\nApply effects, act: ", self.get_name(), " id:", self.get_action_id()," predicates: ", predicates_indexes)
        effects = self._act_effects_list
        #self.converter = self.converter_funct(curr_state, self)#  converter (SympyToTfConverter): The converter instance for SymPy to TensorFlow conversion.
        #out_result=MutableHashTable(key_dtype=tf.string, value_dtype=tf.float32, default_value=MISSING_VALUE)
        out_result=list()
        for effect_indx in effects:
            effect_data=GlobalData._class_effects_list[effect_indx]
            if DEBUG>5:
                print("Applying effect:", effect_data.effect)
            result=1.0
            if effect_data.condition>=0:
                #result = self.evaluate_condition(effect_data.condition, predicates_indexes) 
                result=TensorAction.evaluate_condition(effect_data.condition, self.curr_state, predicates_indexes)
                if DEBUG>4:
                    print("Condition result:", result)
            #if result>0:
            key,value=self.apply_single_effect(effect_indx, predicates_indexes, self.curr_state)
                #if tf.equal(key, metric_expr): # XXX
                #    metric_value_found = True
                #    if self._are_preconditions_satisfied<0:
                #        value = value + UNSAT_PENALTY + UNSAT_PENALTY * (-1.0) * tanh(self._are_preconditions_satisfied)
            if result>0:
                out_result.append((key,value))

        #if self._are_preconditions_satisfied<0:
        #    if metric_value_found==False:
        #        metric_value = self.curr_state.lookup(metric_expr)
        #        metric_value =  metric_value + UNSAT_PENALTY + UNSAT_PENALTY * (-1.0) * tanh(self._are_preconditions_satisfied)
        #        out_result.append((metric_expr, metric_value))
        
        return out_result

    #@tf.function
    def apply_single_effect(self, effect_indx, predicates_indexes, state):
        return TensorAction.apply_single_effect(effect_indx, predicates_indexes, state)


    

class TfAction(TensorAction):
    def __init__(self, problem: up.model.Problem, plan_action: up.plans.plan.ActionInstance, converter, state):
        """
        Initialize the TensorAction with the problem, action, and converter.
        
        Args:
            problem (Problem): The problem containing the objective and other parameters.
            action (Action): The action to apply.
        """
        super().__init__(problem, plan_action, converter, state)
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
        if plan_action._action in GlobalData._class_lifted_actions_object_map:
            self._tf_lifted_action = GlobalData._class_lifted_actions_object_map[plan_action._action]
        else:
            self._tf_lifted_action = TfLiftedAction(problem, plan_action, converter, state)
            GlobalData._class_lifted_actions_object_map[plan_action._action] = self._tf_lifted_action

        self._lifted_act_subs: Dict[up.model.Expression, up.model.Expression] = dict(
                    zip(plan_action.action.parameters, list(plan_action.actual_parameters))
                )
        lifted_predicates_set = self._tf_lifted_action.get_predicates_list()
        predicates_list = []
        predicates_id_list=[]
        pred_matchings = dict()
        pred_matchings_lifted_data = dict()
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
                indexes=GlobalData.insert_predicates_in_map([new_pred.get_name()]) #returns a list of indexes since I provided a list
                indx=indexes[0]
            predicates_id_list.append(indx)

        self._pred_matchings_lifted_data = pred_matchings_lifted_data
        self._pred_matchings = pred_matchings
        self._up_act_free_vars = lifted_predicates_set

        self._predicates_list = predicates_list   
        self._predicates_indexes = tf.constant([t.numpy() for t in predicates_id_list], dtype=tf.int32)
        effects = self._action.effects
        self.act_effects_list=[]
        for effect in effects:
            ef_name=effect.fluent.get_name()
            self.act_effects_list.append(ef_name)

    #@tf.function
    def apply_action(self, predicates_indexes=None, curr_state=None): #: up.tensor.TensorState):    
        """
        Apply TensorfFlow  action to the state if the preconditions are met, and manage the effects.
        """
        update_state= self._tf_lifted_action.apply_action( self._predicates_indexes, curr_state)        
        self.set_are_preconditions_satisfied(self._tf_lifted_action.get_are_preconditions_satisfied())
        return update_state

    def no_tf_apply_action(self, curr_state=None): #: up.tensor.TensorState):    
        """
        Apply TensorfFlow  action to the state if the preconditions are met, and manage the effects.
        """
        return self._tf_lifted_action.apply_action( self._predicates_indexes, curr_state)

    def evaluate_preconditions(self, state=None):
        """
        Evaluates all preconditions and returns whether they are satisfied.
        
        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """
        if DEBUG>5:
            print("Evaluate preconditions, act: ", self.get_name(), " id:", self._action_id)

        if state is not None:
            self.curr_state = state
            self.converter.set_state(state)
  
        satisfied=self._tf_lifted_action.evaluate_preconditions(self._predicates_indexes, self.curr_state)
        return satisfied

    

    #@tf.function
    def apply_effects(self): #up_id useul for tf.function retracing
        """
        Apply the effects of the action to the new state.

        Args:
            effects (list): A list of effect objects to apply.
            new_state (dict): The new state that will be modified.
        """
        if DEBUG>5:
            print("Apply effects, act: ", self.get_name(), " id:", up_id)
     
        out_result=self._tf_lifted_action.apply_effects(self._predicates_indexes)

        return out_result


    def set_curr_state(self, state):
        self.curr_state=state
        self._tf_lifted_action.set_curr_state(state)
        
