from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.math import tanh
import unified_planning as up

from unified_planning.model import OperatorKind
from unified_planning.model import EffectKind

from unified_planning.engines.compilers import Grounder, GrounderHelper
from unified_planning.tensor.converter import SympyToTfConverter
from unified_planning.tensor.tensor_state import TensorState
from unified_planning.tensor.tensor_action import TensorAction, TfLiftedAction, TfAction

from unified_planning.tensor.constants import *

import threading

class TensorPlan(ABC):
    def __init__(self, problem, tensor_state, plan, action_type=up.tensor.TfAction):
        """
        Initializes the TensorPlan object.

        :param problem: The problem instance.
        :param plan: The plan instance containing the actions to be executed.
        :param action_type: The type of action to use ("TfAction" or "TorchAction").
        """
        self.plan = plan
        self.tensor_state = tensor_state
        GlobalData.tensor_state=tensor_state

        self.state_values = None
        
        self.actions_sequence = {}  # Dictionary to store executed actions
        self.apply_funct_actions_sequence_list =[] 
        self.apply_indx_actions_sequence_list =[] 
        self.predicates_actions_sequence_list =[]
        self.effects_pos_actions_sequence_list=[]
        
        self.state_updates = {} # The sequence of states updates after applying actions 
        self.new_action = action_type  # Choose between TfAction and TorchAction

        self.converter_funct = SympyToTfConverter
        self.converter = self.converter_funct(self.tensor_state)

        self.problem = problem
        self.goal_list=[]

       # Update the metric value if the preconditions are not 
        self._problem_metric=self.problem.quality_metrics[0]
        self.str_metric_expr=str(self._problem_metric.expression)
        self.tf_metric_expr=tf.constant(str(self._problem_metric.expression), tf.string)
        self._applicable_actions_list=None

        #self.problem_kind=pk = problem.kind
        #if not Grounder.supports(pk):
        #    msg = f"The Grounder used in the {type(self).__name__} does not support the given problem"
        #    if self.error_on_failed_checks:
        #        raise UPUsageError(msg)
        #    else:
        #        warn(msg)

        # Ensure the action type is valid
        if self.new_action not in [up.tensor.TfAction]: #, up.tensor.TorchAction]:
            raise ValueError("Invalid action type. Use 'TfAction' or 'TorchAction'.")
         
        # Build the sequence of actions
        self.build()


    def build(self):
        """
        Build a sequence of actions on the given state.

        :param my_state: The initial state before applying the actions.
        :return: The resulting state after all actions have been applied.
        """
        problem=self.problem
        tensor_state =self.tensor_state

        if DEBUG>0:
            print("..Building plan")
        
            if DEBUG>2:
                print("Build Initial state: ")
                TensorState.print_filtered_dict_state(tensor_state,["next"])
        
        self.states_updates=None
        for i, act in enumerate(self.plan.actions):

            # Create a grounded action if not already created
            act_name = str(act)
            if act_name in GlobalData._class_grounded_actions_map:
                tensor_action = GlobalData._class_grounded_actions_map[act_name]
            else:
                tensor_action = self.new_action(problem=self.problem, plan_action=act, converter=self.converter, tensor_state=tensor_state)
                GlobalData._class_grounded_actions_map[act_name] = tensor_action

            
            self.actions_sequence[i] = tensor_action
            self.apply_funct_actions_sequence_list.append(tensor_action.apply_action_concrete_funct)
            self.apply_indx_actions_sequence_list.append(tensor_action.get_lifted_action_indx())
            self.predicates_actions_sequence_list.append(tensor_action.get_predicates_indexes())
            self.effects_pos_actions_sequence_list.append(tensor_action.get_effects_pos())

            if DEBUG>1:
                print("Step: ", i, " act name: ", tensor_action.get_name())
                if DEBUG>4:
                    print("Action: ", tensor_action) 
    
        fve = up.model.walkers.FreeVarsExtractor()
        predicates_set= set()

        for goal in self.problem.goals:
                predicates_set.update(fve.get(goal))
                
        self.tensor_state.insert_zero(predicates_set)
        tensor_state.compute_keys_positions()
        for goal in self.problem.goals:
            fluent_name = goal.get_name()

            goal_position = TensorAction.store_condition(goal,self.converter)
            # Store the precondition position to build the action's preconditions_list
            self.goal_list.append(goal_position)

        #Provide a tensor representation of the predicates
        GlobalData.tf_class_predicates_list=tf.stack(GlobalData._class_predicates_list)


        # Ensure the problem has the correct metric defined
        if not any(isinstance(metric, up.model.metrics.MinimizeSequentialPlanLength) or (metric.is_minimize_expression_on_final_state()) for metric in problem.quality_metrics):
            raise ValueError("The problem does not have MinimizeSequentialPlanLength as a quality metric.")
            


    def forward_sequence(self, state_values):
        """
        Executes a sequence of actions on the given state.

        :param my_state: The initial state before applying the actions.
        :return: The resulting state after all actions have been applied.
        """
        self.state_values=state_values
        if DEBUG > 0:
            print(".forward")
            #tf.print("Metric: ", state_values[self.tensor_state.pos_metric_expr])
            if DEBUG>5:
                print("Initial state: ", state_values)
        self._applicable_actions_list=[]

        for tensor_action in self.actions_sequence.values():    
            
            # Apply the action to the current state
            if DEBUG>2:
                tf.print("\nApply Action in ", self._applicable_actions_list.size(), " name: ", tensor_action.get_name())
            are_preconditions_satisfied,indexes,values=self.apply_action(tensor_action.get_lifted_action_indx(), tensor_action.get_predicates_indexes() ,state_values) 
            #are_preconditions_satisfied,indexes,values=tensor_action.apply_action_concrete_funct (tensor_action.get_predicates_indexes(),state_values)
            
            self._applicable_actions_list.append(are_preconditions_satisfied)
     
            for pos,value in zip(indexes.numpy(),values.numpy()):
                if pos>=0:   
                    new_values [pos]=value
                    if DEBUG>5:
                        print("Update: ", self.tensor_state.get_key(int(pos)), ": ", float(value), "new: ", float(state_values[int(pos)]))
                else:
                    if DEBUG>5:
                        print("NUpdate: ", pos, ": ", float(value))
                
            
        if DEBUG>-1:
            valid=self.get_plan_metric(state_values)
            tf.print("Check: ", valid)

            if DEBUG>5:
                print(state_values)
                print()

        quality=state_values[GlobalData.pos_metric_expr]

        return quality

    def forward(self, state_values):
        """
        Executes a sequence of actions on the given state.

        :param my_state: The initial state before applying the actions.
        :return: The resulting state after all actions have been applied.
        """
        self.state_values=state_values
        if DEBUG > 0:
            print(".forward")
            #tf.print("Metric: ", state_values[self.tensor_state.pos_metric_expr])
            if DEBUG>5:
                print("Initial state: ", state_values)
        self._applicable_actions_list=[]

        step=0
        for tensor_action in self.apply_funct_actions_sequence_list:   
            
            # Apply the action to the current state
            are_preconditions_satisfied,values=tensor_action(self.apply_indx_actions_sequence_list[step],self.predicates_actions_sequence_list[step] ,state_values) 
            #are_preconditions_satisfied,values=tensor_action.apply_action_concrete_funct (tensor_action.get_predicates_indexes(),state_values)
            indexes=self.effects_pos_actions_sequence_list  
            self._applicable_actions_list.append(are_preconditions_satisfied)
     
            for pos,value in zip(indexes.numpy(),values.numpy()):
                if pos>=0:   
                    state_values [pos]=value
                    if DEBUG>5:
                        print("Update: ", self.tensor_state.get_key(int(pos)), ": ", float(value), "new: ", float(state_values[int(pos)]))
                else:
                    if DEBUG>5:
                        print("NUpdate: ", pos, ": ", float(value))
                
            
        valid=self.check_goals(state_values)
        if DEBUG>-1:
            valid=self.get_plan_metric(state_values)
            tf.print("Check: ", valid)

            #if DEBUG>5:
            #    print(state_values)
            #    print()

        quality=state_values[GlobalData.pos_metric_expr]

        return quality
    
    def check_goals(self, state_values):
        """
        Check if the goals are satisfied in the current state.

        :return: True if all goals are satisfied, False otherwise.
        """
        metric_value_add = 0.0
        satisfied=1
        for goal in self.goal_list:

            predicates_indexes=GlobalData._class_conditions_list[goal].predicates_indexes
            sympy_function = GlobalData._class_conditions_list[goal].sympy_function  
            value=  sympy_function(predicates_indexes,state_values)
            #sympy_expr = GlobalData._class_conditions_list[goal].sympy_expr   
            #value=  self.converter.convert(sympy_expr, predicates_indexes)
            #value = TensorAction.evaluate_condition(goal)
           
            if value < 0:
                if DEBUG>1:
                    tf.print("Fluent unsatisfied indx: ", goal,", value: ",value)
                satisfied=satisfied*0 # Needed for tf function; use a break?
                metric_value_add +=  UNSAT_PENALTY    *  value *( -1.0 )

        state_values.scatter_nd_add(indices=[[GlobalData.pos_metric_expr]], updates=[metric_value_add])
        
        return satisfied
    

    def get_plan_metric(self, state_values):
        """
        Checks the metric value of a solution plan for a problem that uses MinimizeSequentialPlanLength.

        :param problem: The UPF problem instance.
        :param plan: The UPF SequentialPlan instance.
        :return: A dictionary with validity and the metric value.
        """
        if valid == False:
            tf.print("..Goals NOT satisfied")
        else:
            tf.print("..Goals satisfied")

        # Metric value for MinimizeSequentialPlanLength is the number of actions
        metric_value = state_values[self.tensor_state.pos_metric_expr]
        metric_applicable = state_values[self.tensor_state.pos_are_prec_sat] # len(plan.actions)
        plan_len=len(self.apply_funct_actions_sequence_list)
        #step=0
        #for prec_sat in self._applicable_actions_list:
        #    if prec_sat>=0:
        #        metric_applicable += 1
        #        #if DEBUG > 1:
        #        #    act=self.actions_sequence[step]
        #        #    tf.print("Action in ", step, " is applicable: ", act.get_name())
        #    else:
        #        valid=0
        #        #if DEBUG > 0:
        #        #    act=self.actions_sequence[step]
        #        #    tf.print("Action in ", step, " is NOT applicable: ", act.get_name())
        #    step+=1
                
        return {
            "valid": valid,
            "applicable": metric_applicable,
            "metric_value": metric_value,
            "steps": plan_len,
        }
    
    def get_final_state(self):
        return self.state_values
    
    def get_values(self):
        """
        Get the resulting state after executing the plan.

        :return: The resulting state after executing the plan.
        """
        return self.state_values
 



class TfPlan(TensorPlan):
    def __init__(self, problem, tensor_state, plan):
        """
        Initializes the TfPlan object, which is a specific type of TensorPlan using TfAction.

        :param problem: The problem instance.
        :param plan: The plan instance containing the actions to be executed.
        """
        super().__init__(problem, tensor_state, plan, action_type=up.tensor.TfAction)

    #@tf.function
    def forward(self, state_values):
        """
        Executes a sequence of actions on the given state.

        :param my_state: The initial state before applying the actions.
        :return: The resulting state after all actions have been applied.
        """
        self.state_values=state_values
        #if DEBUG > 0:
        #    print(".forward")
        #    tf.print("Metric: ", state_values[self.tensor_state.pos_metric_expr])
        #    if DEBUG>5:
        #        print("Initial state: ", state_values)
        self._applicable_actions_list=[]
        max_size=len(self.apply_funct_actions_sequence_list)
        for step in range(max_size):    
            
            # Apply the action to the current state
            #if DEBUG>2:
            #    tf.print("\nApply Action in ", self._applicable_actions_list.size(), " name: ", tensor_action.get_name())
            predicates_list=self.predicates_actions_sequence_list[step]
            apply_action=self.apply_funct_actions_sequence_list[step]
            indexes=self.effects_pos_actions_sequence_list[step]
            #are_preconditions_satisfied,state_update=tensor_action.apply_action(state_values)
            #are_preconditions_satisfied,state_update=tensor_action.apply_action_funct (tensor_action.get_predicates_indexes(),state_values)
            #are_preconditions_satisfied,indexes,values=TfLiftedAction.apply_TfLiftedAction(tensor_action.get_lifted_action_indx(), tensor_action.get_predicates_indexes() ,state_values) 
            are_preconditions_satisfied,values=apply_action(predicates_list,state_values)

            self._applicable_actions_list.append(are_preconditions_satisfied)
     
            if tf.shape(indexes)[0] > 0:  # Apply updates in batch, we update here the state_values tensor and not in apply_action in order to use input_shape in tf.function
                state_values.scatter_nd_add(indices=indexes, updates=values)
            
            #tf.print("Updates:",indexes, " --- ", values)
            
            #for pos,value in zip(indexes,values):
            #    if pos>=0:   
            #        #new_values.scatter_nd_add(indices=[[pos]], updates=[value])
            #        if DEBUG>-5:
            #            tf.print("Update: ", pos," -- ", self.tensor_state.get_key(int(pos)), ": ", float(value), "new: ", float(state_values[int(pos)]))
            #    else:
            #        tf.print("NUpdate: ", pos, ": ", float(value))
                
            
        goals_valid=self.check_goals(state_values)
        if DEBUG>1:
            check_valid=self.get_plan_metric(state_values)
            tf.print("Check: ", check_valid)

            if DEBUG>5:
                print(state_values)
                print()
        prec_satified= max_size==state_values[GlobalData.pos_are_prec_sat] #tf.cond(max_size==state_values[GlobalData.pos_are_prec_sat], lambda: TF_SAT, lambda: TF_UN_SAT)
        return state_values[GlobalData.pos_metric_expr],state_values[GlobalData.pos_are_prec_sat],prec_satified,goals_valid