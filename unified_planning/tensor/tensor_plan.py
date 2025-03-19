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
            if DEBUG>1:
                print("Step: ", i, " act name: ", tensor_action.get_name())
                if DEBUG>4:
                    print("Action: ", tensor_action) 
    

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
            
       
    #@tf.function
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
        step=0
        self._applicable_actions_list=[]
        for tensor_action in self.actions_sequence.values():    
            
            # Apply the action to the current state
            if DEBUG>2:
                print("\nApply Action in ", step, " name: ", tensor_action.get_name())
                #tf.print("Step: ", step, " storage: ", state_values[1222])
        
            #are_preconditions_satisfied,state_update=tensor_action.apply_action(state_values)
            are_preconditions_satisfied=TfLiftedAction.apply_lifted_action(tensor_action.get_lifted_action_indx(), tensor_action.get_predicates_indexes() ,state_values) 
            self._applicable_actions_list.append((step, are_preconditions_satisfied))
            new_values= state_values
            
            #for pos,value in state_update:
            #    if pos>=0:
            #        orig=state_values[pos]     
            #        new_values.scatter_nd_add(indices=[[pos]], updates=[value])
            #        if DEBUG>5:
            #            tf.print("Update: ", self.tensor_state.get_key(pos), ": ", float(value), "orig: ",float(orig), "new: ", float(new_values[pos]))
           
            metric_value_add = UNSAT_PENALTY + UNSAT_PENALTY * (-1.0) * tanh(are_preconditions_satisfied)
            if tf.math.less(are_preconditions_satisfied, TF_ZERO):
                #print("Action in ", step, " is NOT applicable: ", tensor_action.get_name()) 
                pos=self.tensor_state.pos_metric_expr
                
                if pos>=0:
                    #state_update.append((pos, metric_value_add))
                    new_values.scatter_nd_add(indices=[[pos]], updates=[metric_value_add])
                    #metric_value_add=metric_value_add+new_values[pos]
                    #new_values[pos].assign(metric_value_add)
                if DEBUG>3:  
                    tf.print("Action in ", step, " is NOT applicable: ", tensor_action.get_name(), " penalty: ", metric_value_add, "final: ", new_values[pos], " prec sat: ", are_preconditions_satisfied, " prec sat state: ",new_values[self.tensor_state.pos_are_prec_sat])   

            self.new_values = new_values
            state_values = new_values  # Update the reference for the next iteration

            # Update the new state after applying the action
            step+=1
            
        if DEBUG>-1:
            valid=self.get_plan_metric(new_values)
            tf.print("Check: ", valid)

            if DEBUG>5:
                print(new_values)
                print()

        quality=new_values[self.tensor_state.pos_metric_expr]

        return quality
        
        
    
    def check_goals(self, state_values):
        """
        Check if the goals are satisfied in the current state.

        :return: True if all goals are satisfied, False otherwise.
        """
        
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
                metric_value_add =  UNSAT_PENALTY  #  *  current_value
                state_values.scatter_nd_add(indices=[[self.tensor_state.pos_metric_expr]], updates=[metric_value_add])
             
                #metric_value = state_values[self.tensor_state.pos_metric_expr]+ UNSAT_PENALTY  #  *  current_value
                #state_values[self.tensor_state.pos_metric_expr].assign(metric_value)
     

        return satisfied
    

    def get_plan_metric(self, state_values):
        """
        Checks the metric value of a solution plan for a problem that uses MinimizeSequentialPlanLength.

        :param problem: The UPF problem instance.
        :param plan: The UPF SequentialPlan instance.
        :return: A dictionary with validity and the metric value.
        """
        plan=self.plan
        valid=self.check_goals(state_values)
        if valid == False:
            tf.print("..Goals NOT satisfied")
        else:
            tf.print("..Goals satisfied")

        # Metric value for MinimizeSequentialPlanLength is the number of actions
        metric_value = state_values[self.tensor_state.pos_metric_expr]
        metric_applicable = 0 # len(plan.actions)
        plan_len=len(self.actions_sequence)
        for step,prec_sat in self._applicable_actions_list:
            if prec_sat>=0:
                metric_applicable += 1
                if DEBUG > 1:
                    act=self.actions_sequence[step]
                    tf.print("Action in ", step, " is applicable: ", act.get_name())
            else:
                valid=0
                if DEBUG > 0:
                    act=self.actions_sequence[step]
                    tf.print("Action in ", step, " is NOT applicable: ", act.get_name())
                
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
        return super().forward(state_values)

