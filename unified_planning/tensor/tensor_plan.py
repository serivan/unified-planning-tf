from abc import ABC, abstractmethod
import tensorflow as tf
import unified_planning as up

from unified_planning.model import OperatorKind
from unified_planning.model import EffectKind

from unified_planning.engines.compilers import Grounder, GrounderHelper
from unified_planning.tensor.converter import SympyToTfConverter
from unified_planning.tensor.tensor_state import TensorState
from unified_planning.tensor.tensor_action import TensorAction

from unified_planning.tensor.constants import *

import threading

class TensorPlan(ABC):
    def __init__(self, problem, state, plan, action_type=up.tensor.TfAction):
        """
        Initializes the TensorPlan object.

        :param problem: The problem instance.
        :param plan: The plan instance containing the actions to be executed.
        :param action_type: The type of action to use ("TfAction" or "TorchAction").
        """
        self.plan = plan
        self.tensor_state = state
        self.initial_state=state.get_initial_hash_state()
        self.curr_state =state.get_hash_state()
        self.actions_sequence = {}  # Dictionary to store executed actions
        self.new_state = {}     # The resulting state after applying actions
        self.state_updates = {} # The sequence of states updates after applying actions 
        self.new_action = action_type  # Choose between TfAction and TorchAction

        self.converter_funct = SympyToTfConverter
        self.converter = self.converter_funct(self.curr_state)

        self.problem = problem
        self.goal_list=[]

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
        if DEBUG>0:
            print("..Building plan")
        #curr_state =self.create_copy_state(my_state)
        
        curr_state =self.curr_state

        if DEBUG>2:
            print("Build Initial state: ")
            TensorState.print_filtered_hash_state(curr_state,["next"])
        all_keys=[]
        self.states_updates=None
        for i, act in enumerate(self.plan.actions):
            tensor_action = self.new_action(problem=self.problem, plan_action=act, converter=self.converter, state=curr_state)
            self.actions_sequence[i] = tensor_action
            if DEBUG>1:
                print("Step: ", i, " act name: ", tensor_action.get_name())
            if DEBUG>4:
                print("Action: ", tensor_action) 
            
            tensor_action.set_curr_state(curr_state)
            
            #GlobalData.tf_class_predicates_list=tf.stack(GlobalData._class_predicates_list)
            #new_state= curr_state 
            ## Apply the action to the current state
            #state_update=tensor_action.no_tf_apply_action()

            ## Retrieve all keys and values from state_update    
            #keys = state_update.export()[0]  # Extract keys
            #values = state_update.export()[1]  # Extract values
            keys=tensor_action.act_effects_list
            for key in keys:
                exists = tf.reduce_any(tf.equal(key, all_keys))
                if not exists:
                    all_keys.append(key)

            # Merge state_update into new_state
            #new_state.insert(keys, values)


            # Update the new state after applying the action
            #tensor_action.set_new_state(new_state)
            #self.state_updates[i+1] = state_update
            #self.new_state = new_state
            #curr_state = new_state  # Update the reference for the next iteration

            #if DEBUG > -3:
            #    print("State updates:", end=":: ")
            #    TensorState.print_filtered_hash_state(state_update,["next"])
            #if DEBUG>5:
            #    print("State after action:", end=":: ")
            #    TensorState.print_filtered_hash_state(new_state,["next"])
            #print("New state:", new_state)
        

        for goal in self.problem.goals:
            fluent_name = goal.get_name()

            goal_position = TensorAction.store_condition(goal, self.converter)
            # Store the precondition position to build the action's preconditions_list
            self.goal_list.append(goal_position)

        #Provide a tensor representation of the predicates
        GlobalData.tf_class_predicates_list=tf.stack(GlobalData._class_predicates_list)
        self.tensor_state.restore_hash_state(all_keys)
       

    def forward(self, dict_state):
        """
        Executes a sequence of actions on the given state.

        :param my_state: The initial state before applying the actions.
        :return: The resulting state after all actions have been applied.
        """
        if DEBUG > 0:
            print(".forward")
        curr_state = self.tensor_state.get_hash_state()

        self.state_updates[0]={}
        
        for fluent, value in dict_state.items():
            #print("Fluent:", fluent)
            #print("Initial value:", value)
            curr_state.insert(tf.constant(fluent,tf.string),value)
    
        if DEBUG>2:
            print("Initial state: ", curr_state)
            #print(tf.py_function(func=TensorState.print_filtered_hash_state,inp=(curr_state,["next"]),Tout=tf.string))
        #all_keys=[]
        step=0
        for tensor_action in self.actions_sequence.values():    
            #print("Action: ", tensor_action)
            #self.copy_state_in_new_state(curr_state, new_state)    
            # Apply the action to the current state
            if DEBUG>2:
                print("\nApply Action in ", step, " name: ", tensor_action.get_name())
            tensor_action.set_curr_state(curr_state)

            tensor_action.converter.set_state(self.curr_state)
        
            new_state= curr_state
            state_update=tensor_action.apply_action() 
            if DEBUG >2:
                print("State update: ", state_update)   
            
            # Retrieve all keys and values from state_update    
            keys = state_update.export()[0]  # Extract keys
            values = state_update.export()[1]  # Extract values

            #for key in keys:
            #    exists = tf.reduce_any(tf.equal(key, all_keys))
            #    if not exists:
            #        all_keys.append(key)

            # Merge state_update into new_state
            new_state.insert(keys, values)


            # Update the new state after applying the action
            tensor_action.set_new_state(new_state)
            self.state_updates[step+1] = state_update
            self.new_state = new_state
            curr_state = new_state  # Update the reference for the next iteration

            # Update the new state after applying the action
            step+=1
            
        valid=self.get_plan_metric(new_state)
        if DEBUG>-1:
            tf.print("Check: ", valid)

        if DEBUG>5:
            print(new_state)
            print()

        quality=new_state[str(self.problem.quality_metrics[0].expression)]
        
        self.tensor_state.restore_hash_state()

        return quality
        #return valid["metric_value"]
        
    
    def check_goals(self, state):
        """
        Check if the goals are satisfied in the current state.

        :return: True if all goals are satisfied, False otherwise.
        """
        self.converter.set_state(state)
        
        metric=self.problem.quality_metrics[0]
        metric_expr=str(metric.expression)
        satisfied=1
        for goal in self.goal_list:
            value = TensorAction.evaluate_condition(goal, self.converter)

            if value < 0:
                tf.print("Fluent unsatisfied indx: ", goal,", value: ",value)
                satisfied=satisfied*0 # Needed for tf function; use a break?
     
                metric_value = state[metric_expr]+ UNSAT_PENALTY  #  *  current_value
                state.insert(metric_expr,metric_value)
     

        return satisfied
    

    def get_plan_metric(self, state):
        """
        Checks the metric value of a solution plan for a problem that uses MinimizeSequentialPlanLength.

        :param problem: The UPF problem instance.
        :param plan: The UPF SequentialPlan instance.
        :return: A dictionary with validity and the metric value.
        """
        problem=self.problem
        plan=self.plan
        valid=self.check_goals(state)
        if valid == False:
            tf.print("..Goals NOT satisfied")
        else:
            tf.print("..Goals satisfied")

        # Ensure the problem has the correct metric defined
        if not any(isinstance(metric, up.model.metrics.MinimizeSequentialPlanLength) or (metric.is_minimize_expression_on_final_state()) for metric in problem.quality_metrics):
            return {
                "valid": False,
                "error": "The problem does not have MinimizeSequentialPlanLength as a quality metric.",
            }
        # Metric value for MinimizeSequentialPlanLength is the number of actions
        metric=problem.quality_metrics[0]
        metric_value = state[str(metric.expression)]
        metric_applicable = 0 # len(plan.actions)
        for step,act in self.actions_sequence.items():
            if act.is_applicable() == True:
                metric_applicable += 1
                if DEBUG > 1:
                    tf.print("Action in ", step, " is applicable: ", act.get_name())
            else:
                valid=0
                if DEBUG > 0:
                    tf.print("Action in ", step, " is NOT applicable: ", act.get_name())
                
        return {
            "valid": valid,
            "applicable": metric_applicable,
            "metric_value": metric_value,
            "steps": len(plan.actions),
        }
    
    def get_final_state(self):
        return self.get_state()
    
    def get_state(self):
        """
        Get the resulting state after executing the plan.

        :return: The resulting state after executing the plan.
        """
        return self.new_state
 



class TfPlan(TensorPlan):
    def __init__(self, problem, state, plan):
        """
        Initializes the TfPlan object, which is a specific type of TensorPlan using TfAction.

        :param problem: The problem instance.
        :param plan: The plan instance containing the actions to be executed.
        """
        super().__init__(problem, state, plan, action_type=up.tensor.TfAction)


    #@tf.function
    def forward(self, dict_state):
        """
        Executes a sequence of actions on the given state.

        :param my_state: The initial state before applying the actions.
        :return: The resulting state after all actions have been applied.
        """
        return super().forward(dict_state)
