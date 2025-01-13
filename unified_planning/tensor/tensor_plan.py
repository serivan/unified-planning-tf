from abc import ABC, abstractmethod
import tensorflow as tf
import unified_planning as up

from unified_planning.model import OperatorKind
from unified_planning.model import EffectKind

from unified_planning.engines.compilers import Grounder, GrounderHelper
from unified_planning.tensor.converter import SympyToTfConverter
from unified_planning.tensor.tensor_state import TensorState

from unified_planning.tensor.constants import *

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
        self.initial_state = self.curr_state =state.get_tensors()
        self.actions_sequence = {}  # Dictionary to store executed actions
        self.new_state = {}     # The resulting state after applying actions
        self.states_sequence = {} # The sequence of states after applying actions 
        self.action_type = action_type  # Choose between TfAction and TorchAction

        self._converter_funct = SympyToTfConverter
        self.converter = self._converter_funct(state)

        self.problem = problem
        #self.problem_kind=pk = problem.kind
        #if not Grounder.supports(pk):
        #    msg = f"The Grounder used in the {type(self).__name__} does not support the given problem"
        #    if self.error_on_failed_checks:
        #        raise UPUsageError(msg)
        #    else:
        #        warn(msg)

        # Ensure the action type is valid
        if self.action_type not in [up.tensor.TfAction]: #, up.tensor.TorchAction]:
            raise ValueError("Invalid action type. Use 'TfAction' or 'TorchAction'.")
         
        # Build the sequence of actions
        self.build(self.initial_state)


    def build(self, my_state):
        """
        Build a sequence of actions on the given state.

        :param my_state: The initial state before applying the actions.
        :return: The resulting state after all actions have been applied.
        """
        if DEBUG>0:
            print("Building plan")
        #curr_state =self.create_copy_state(my_state)
        curr_state =TensorState.shallow_copy_dict_state(my_state)

        self.states_sequence[0] = curr_state
        
        for i, act in enumerate(self.plan.actions):
            tensor_action = self.action_type(problem=self.problem, plan_action=act, converter_funct=self._converter_funct)
            self.actions_sequence[i] = tensor_action
            if DEBUG>1:
                print("Step: ", i, " act name: ", tensor_action.get_name())
            if DEBUG>4:
                print("Action: ", tensor_action) 
            
            # Apply the action to the current state
            new_state=tensor_action.apply_action(curr_state)
            self.states_sequence[i+1] = new_state
            self.new_state = new_state
            curr_state = new_state  # Update the reference for the next iteration

            #print("New state:", new_state)
       

    def forward(self, dict_state):
        """
        Executes a sequence of actions on the given state.

        :param my_state: The initial state before applying the actions.
        :return: The resulting state after all actions have been applied.
        """
        if DEBUG > 0:
            print(".forward")
        curr_state = self.states_sequence[0]
        
        #self.reset_states_sequence(self.initial_state)
        
        for fluent, value in dict_state.items():
            #print("Fluent:", fluent)
            #print("Initial value:", value)
            curr_state[fluent].assign(value)
    
        step=0
        for tensor_action in self.actions_sequence.values():    
            #print("Action: ", tensor_action)
            #self.copy_state_in_new_state(curr_state, new_state)    
            # Apply the action to the current state
            if DEBUG>2:
                print("Apply Action in ", step, " name: ", tensor_action.get_name())
            new_state=tensor_action.apply_action(curr_state)
            # Update the new state after applying the action
            step+=1
            self.new_state = new_state
            curr_state = new_state
            self.states_sequence[step] = new_state
            #print("New state:", self.new_state)
            
        valid=self.get_plan_metric(new_state)
        if DEBUG>=0:
            tf.print("Check: ", valid)

        if DEBUG>5:
            print(new_state)
            print()
        return new_state[str(self.problem.quality_metrics[0].expression)]
        #return valid["metric_value"]
        
    
    def check_goals(self, state):
        """
        Check if the goals are satisfied in the current state.

        :return: True if all goals are satisfied, False otherwise.
        """
        self.converter.set_state(state)
        
        satisfied=1
        for goal in self.problem.goals:
            fluent_name = goal.get_name()
            if goal.node_type != OperatorKind.FLUENT_EXP:
                value = self.converter.compute_condition_value(goal)
                current_value = value
            else:
                current_value = state[fluent_name]

            if current_value <= 0:
                tf.print("Fluent unsatisfied: ", fluent_name,", value: ",current_value)
                satisfied=satisfied*0 # Needed for tf function; use a break?

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
        #for step,act in self.actions_sequence.items():
        #    if act.is_applicable() <= 0:
        #        tf.print("Action ", step," not applicable:", act.get_name(), act._are_preconditions_satisfied)
        for step, tensor_action in list(self.actions_sequence.items())[:-1]:

            if tensor_action.get_are_preconditions_satisfied() > 0:
                if DEBUG > 1:
                    tf.print("Action in ", step, " is applicable: ", tensor_action.get_name())
                metric_applicable += 1
            else:
                if DEBUG > 0:
                    tf.print("Action in ", step, " is NOT applicable: ", tensor_action.get_name())
                    
                
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
 

    def reset_dict_states_sequence(self, original_state):
        """ 
        Reset the states sequence to the original state.
        """
        assign_ops = []
        for state in self.states_sequence.values():
            #Creare un elenco di operazioni di assegnazione (assign_ops
            assign = [state[key].assign(original_state[key]) for key in original_state]
            assign_ops.append(assign)

        # Raggruppare tutte le operazioni in una singola operazione batch
        assign_batch_op = tf.group(*assign_ops)

        # Esegui tutte le operazioni di copia in un colpo solo
        assign_batch_op


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