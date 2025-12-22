
# Commented out IPython magic to ensure Python compatibility.
#!apt-get install openjdk-17-jdk
# %pip install unified-planning[enhsp]
#%pip install unified-planning

"""We are now ready to use the Unified-Planning library!

but first import libraries for pytorch
"""

# Commented out IPython magic to ensure Python compatibility.
## Standard libraries
import os
import math
import numpy as np
import time

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
#int = lambda value: tf.constant(value, dtype=tf.Tensor)
import tensorboard
from tensorboard import main as tb
from tensorboard import default
from tensorboard import program

import traceback
import contextlib

# Clear any logs from previous runs
import os, shutil


import sympy as sp
import os

import datetime

"""

We start importing the shortcuts.
"""

from unified_planning.shortcuts import *
from unified_planning.test import TestCase

from pickle import NONE
from unified_planning.engines import UPSequentialSimulator, SequentialSimulatorMixin
from unified_planning.model import State
from unified_planning.plans import ActionInstance
from unified_planning.test import unittest_TestCase, main
from unified_planning.test.examples import get_example_problems
from unified_planning.exceptions import UPUsageError
from unified_planning.tensor.tensor_action import TfAction
from unified_planning.tensor.tensor_plan import TensorPlan, TfPlan
from unified_planning.tensor.tensor_state import TensorState, TfState
from unified_planning.io import PDDLWriter, PDDLReader
from unified_planning.model.metrics import MinimizeSequentialPlanLength
from unified_planning.model.metrics import MinimizeExpressionOnFinalState


from unified_planning.plans import SequentialPlan

from unified_planning.tensor.constants import *
#Profiling

from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput


"""UTILS"""
EPSILON=1e-7  #tf.keras.backend.epsilon()
SOL_FILE="result_plan.sol"
SOL_FILE="r.sol"


"""#Tensorboard"""


# Load tensorboard extension for Jupyter Notebook, only need to start TB in the notebook
# %load_ext tensorboard

tensorboard.__version__

"""Define a helper function to demonstrate the kinds of errors you might encounter:"""

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))
  

path='./logs/'
shutil.rmtree(path)
#!rm -rf 

# Define a log directory with a timestamp to avoid overwrites
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

tracking_address = "./logs/fit/" # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    #url = tb.launch()
    #print(f"Tensorflow listening on {url}")



# Planning Demo
############################################################
# Initialize the planning problem
problem = Problem("WaterTransferProblem")

# Define fluent state variables
large_container = Fluent("large_container", RealType())
small_container = Fluent("small_container", RealType())

cost = Fluent("cost", RealType())
# Define actions
fill_largei = InstantaneousAction("fill_largei", amount=RealType())
amount = fill_largei.parameters

#fill_large.add_effect(large_container,Plus(large_container, small_container))

fill_largei.add_increase_effect(cost,Plus(small_container, amount))
#fill_largei.add_increase_effect(cost,Times(large_container, small_container))
fill_largei.add_increase_effect(small_container, amount)
#fill_largei.add_decrease_effect(small_container,amount)

fill_largei.add_increase_effect(cost, small_container)

#fill_largei.add_decrease_effect(large_container, amount)
# Example precondition: Only fill if large_container is empty
fill_largei.add_precondition(LT(large_container, 80))
##fill_largei.add_precondition(GE(amount, 30))
#fill_largei.add_precondition(GE(small_container, amount))
#fill_largei.add_precondition(LT(Times(large_container, small_container), 800))
fill_largei.add_precondition(GE(small_container, 0))
fill_largei.add_precondition(GE(amount, 0))

# Add conditional effect
fill_largei.add_effect( cost, Plus(cost, amount), GT(small_container, 10)) 

# Define actions
fill_large_neg = InstantaneousAction("fill_large_neg", amount=RealType())
amount = fill_large_neg.parameters

#fill_large.add_effect(large_container,Plus(large_container, small_container))

fill_large_neg.add_increase_effect(cost,Plus(small_container, amount))
#fill_large_neg.add_increase_effect(cost,Times(large_container, small_container))
fill_large_neg.add_decrease_effect(small_container, 50)
#fill_large_neg.add_decrease_effect(small_container,amount)

fill_large_neg.add_increase_effect(cost, small_container)

#fill_large_neg.add_decrease_effect(large_container, amount)
# Example precondition: Only fill if large_container is empty
fill_large_neg.add_precondition(LT(large_container, 80))
##fill_large_neg.add_precondition(GE(amount, 30))
#fill_large_neg.add_precondition(GE(small_container, amount))
#fill_large_neg.add_precondition(LT(Times(large_container, small_container), 800))
fill_large_neg.add_precondition(LT(small_container, 10))
fill_large_neg.add_precondition(GE(amount, 0))

# Add conditional effect
fill_large_neg.add_effect( cost, Plus(cost, amount), GT(small_container, 10)) 

# Define actions
#fill_large = InstantaneousAction("fill_large", amount=RealType())
fill_larged = DurativeAction("fill_large", amount=RealType())
amount = fill_larged.parameters
fill_larged.set_fixed_duration(amount)
#fill_larged.add_effect(large_container,Plus(large_container, small_container))
fill_larged.add_increase_effect(EndTiming(),cost,Times(large_container, small_container))
fill_larged.add_increase_effect(EndTiming(),cost, amount)
#fill_larged.add_increase_effect(cost, small_container)

fill_larged.add_decrease_effect(EndTiming(),small_container,amount)

fill_larged.add_increase_effect(EndTiming(),large_container, amount)
# Example precondition: Only fill if large_container is empty
fill_larged.add_condition(StartTiming(), LT(large_container, 80))
fill_larged.add_condition(StartTiming(), GE(large_container, 0))
fill_larged.add_condition(StartTiming(), LT(Times(large_container, small_container), 800))

# Add conditional effect
#fill_larged.add_condition(StartTiming(),  cost, Plus(cost, large_container), GT(small_container, 10)) 


# Add fluents and actions to the problem
problem.add_fluent(large_container)
problem.add_fluent(small_container)
problem.add_fluent(cost)
problem.add_action(fill_largei)
#problem.add_action(fill_large_neg)
#problem.add_action(fill_larged)
#problem.set_initial_value(large_container, tf.constant(20))
problem.set_initial_value(large_container, 70)
problem.set_initial_value(small_container, 5)

problem.set_initial_value(cost, 0)

#amount = Fluent("amount", RealType())
#problem.set_initial_value(amount, 5)
#problem.add_fluent(amount)


#NEW TEST PROBLEM
#problem = Problem("WaterTransferProblem")

#Propositional effects
Location = UserType('Location')
robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
connected = unified_planning.model.Fluent('connected', BoolType(), l_from=Location, l_to=Location)
move = InstantaneousAction('move', l_from=Location, l_to=Location)
l_from = move.parameter('l_from')
l_to = move.parameter('l_to')
move.add_precondition(connected(l_from, l_to))
move.add_precondition(robot_at(l_from))
move.add_precondition(LT(large_container, 80))
move.add_precondition(GE(large_container, 0))
move.add_precondition(GE(small_container, 30))

move.add_increase_effect(cost, large_container)
move.add_increase_effect(cost, small_container)
move.add_effect(robot_at(l_from), False)
move.add_effect(robot_at(l_to), True)
move.add_decrease_effect(large_container,5)

tf.print(move)
problem.add_fluent(robot_at, default_initial_value=False)
problem.add_fluent(connected, default_initial_value=False)
problem.add_action(move)

NLOC = 10
locations = [unified_planning.model.Object('l%s' % i, Location) for i in range(NLOC)]
problem.add_objects(locations)

problem.set_initial_value(robot_at(locations[0]), True)
for i in range(NLOC - 1):
    problem.set_initial_value(connected(locations[i], locations[i+1]), True)

problem.add_goal(robot_at(locations[1]))
#problem.add_goal(robot_at(locations[-1]))
#problem.add_goal(GE(small_container, 30))


#metric = MinimizeSequentialPlanLength()
metric=MinimizeExpressionOnFinalState(cost)
problem.add_quality_metric(metric)
tf.print("Prolem: ",problem)


# Define the file path
domain_file = "domain.pddl"
problem_file = "problem.pddl"

# Create a writer and export the problem
writer = PDDLWriter(problem)
#writer.write_domain(domain_file)
writer.write_problem(problem_file)

# Solve the planning problem
sol_plan=None
if not os.path.exists(SOL_FILE):

  with OneshotPlanner(name='pyperplan') as planner:
    result = planner.solve(problem)
    if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
        tf.print("Pyperplan returned: %s" % result.plan)
        # Save result.plan to a file
        writer = PDDLWriter(problem)
        writer.write_plan(result.plan, "result_plan.sol")
    else:
        tf.print("No plan found.")


# Set the initial state
#initial_state = {large_container:  tf.constant(1.0), small_container:  tf.constant(0, dtype=tf.float32)}
initial_state ={}

tensor_state=TfState(problem)
if False and os.path.exists(SOL_FILE):
  # Reload the saved plan from the file
  # Reload the saved PDDL solution file
  reader = PDDLReader()
  sol_plan = reader.parse_plan(problem,SOL_FILE)
  tf.print("Plan: ", sol_plan)

#  Insert the manually defined plan
sol_plan = SequentialPlan([ActionInstance(fill_largei, 500.0),ActionInstance(fill_large_neg, 500.0),
                           ActionInstance(move, (locations[0], locations[1])),
    ]) 
    #,    ActionInstance(fill_largei, 200.0)])

def change_initial_state(plan, initial_state):
  state_values=plan.tensor_state.get_initial_state_values()
  for fluent, value in initial_state.items():
        #tf.print("Fluent:", fluent)
        #tf.print("Initial value:", value)

        pos=plan.tensor_state.get_key_position(fluent)
        if pos>=0:
            state_values[pos].assign(value)
            tf.print("Fluent: ", fluent, " value: ", state_values[pos])

  
  return state_values

def generate_variables_values(plan):
  variables_values = []
  for act in plan.actions:
      tf_action = TfAction(act.action, plan.tensor_state)
      act_vars = tf_action.generate_variables_values()
      variables_values.extend(act_vars)

  plan.variables= tf.Variable(variables_values, dtype=tf.float32, trainable=True)

@tf.function(
    reduce_retracing=True,
    experimental_relax_shapes=True
)
def plan_sequence(initial_state, plan, learning_rate=0.1, steps=1000):
  steps = tf.convert_to_tensor(steps, dtype=tf.int32)

  seq_plan = plan  # assuming `plan` is actually a TensorPlan object
  state = change_initial_state(seq_plan, initial_state)
  
  # Assume: one tf.Variable per action (e.g. plan.variables = [w0, w1, ..., wN])
  # Using tuple is slightly friendlier for tf.function than a Python list that changes.
  variables_values = tuple(plan.variables)

  # These will hold values from the last iteration for logging/return
  loss= tf.constant(0.0, dtype=tf.float32)
  are_prec_sat= tf.constant(0.0, dtype=tf.float32)
  goals_valid= tf.constant(0, dtype=tf.int32)

  # Variables to keep track of the "best so far"
  best_loss = tf.constant(float("inf"), dtype=tf.float32)
  # Best snapshot of each variable; tensors, not Variables
  best_vars = [tf.identity(v) for v in variables_values]

  #tf.print("===================================================")
  #tf.print("Orig Vars:", variables_values)
  
  for step in tf.range(steps):
    state, are_prec_sat, goals_valid, loss, clipped_grads = execute_step(state, plan, variables_values)

    # SGD update per ogni azione/peso
    for v, g in zip(variables_values, clipped_grads):
      v.assign_sub(learning_rate * g)

  
    # Check if this step has a better loss than any previous one
    is_better = loss < best_loss

    # Branch that *replaces the entire snapshot* when loss improves
    def update_best():
        # Copy the full content of each variable
        new_best_vars = [tf.identity(v) for v in variables_values]
        new_best_loss = loss
        return new_best_loss, new_best_vars

    def keep_best():
        # Keep previous best_loss and best_vars unchanged
        return best_loss, best_vars

    # Use tf.cond so that either we fully update all best_vars
    # or we keep them as they are (no mixing per-element).
    best_loss, best_vars = tf.cond(is_better, update_best, keep_best)


  # After the loop: restore the best configuration of variables_values
  for v, best_v in zip(variables_values, best_vars):
      v.assign(best_v)

  # DEBUG finale (solo dopo l’ultima iterazione)
  tf.print("Step", steps - 1, "- Loss:", loss)
  tf.print("Prec sat:", are_prec_sat)
  tf.print("Goals valid:", goals_valid)
  tf.print("Best loss:", best_loss)

  # If you really need final variable values / state, log them once here.
  # Avoid per-element loops; this keeps the compiled graph small and faster.
  tf.print("Final variables:", variables_values)
  tf.print("Final state (first 10 elements):", state[:10])

  # Stampa valori finali dei "pesi" per azione
  #for i, (name, v) in enumerate(zip(GlobalData._class_variables_list, variables_values)):
  #  tf.print("Variable", i, "name:", name, "value:", v)

  # Stampa stato finale
  #for i in range(seq_plan.tensor_state.size()):
  #  tf.print("key:", seq_plan.tensor_state.get_key(i), "=", state[i])

  tf.print("===================================================")      
     
  
  return loss


@tf.function
def execute_step(state, plan, variables_values, clip_norm=10.0):
    """
    Single optimization step:
    - Performs a forward pass through all actions in the plan.
    - Accumulates a total loss.
    - Applies goal checking on the final state.
    - Computes gradients w.r.t. all variables_values.
    - Applies global-norm gradient clipping and returns clipped gradients.
    """

    #return state, 1, 1, tf.constant(0.0), variables_values
    with tf.GradientTape() as tape:
        # Watch all variables (one per action)
        tape.watch(variables_values)

        current_state = state
        total_loss = tf.constant(0.0, dtype=tf.float32)
        are_prec_sat = tf.constant(1, dtype=tf.int32)  # example default

        # Forward pass through all actions, like a chain of layers
        for action_vars in variables_values:
            # plan.forward_step is assumed to have the same interface as before:
            #   loss_i, are_prec_sat, new_state = plan.forward_step(state, vars)
            loss_i, are_prec_sat, current_state = plan.forward_step(
                current_state, action_vars
            )
            # Aggregate the loss (here we sum, but you can change aggregation if needed)
            total_loss = total_loss + loss_i

        # Goal checking on the final state
        goals_valid, metric_value_add = plan.check_goals(
            current_state, variables_values[-1]
        )

        if goals_valid <= 0:
            # Update the metric in the state and use it as the final loss
            state_values = tf.tensor_scatter_nd_add(
                current_state,
                indices=[[GlobalData.pos_metric_expr]],
                updates=[metric_value_add],
            )
            total_loss = tf.gather(state_values, GlobalData.pos_metric_expr)

    # Compute gradients with respect to all action variables
    grads = tape.gradient(total_loss, variables_values)

    # Post-process gradients (handle None and IndexedSlices)
    processed_grads = []
    for v, g in zip(variables_values, grads):
        # If gradient is None, replace with zeros of the same shape
        if g is None:
            g = tf.zeros_like(v)
        # Convert IndexedSlices to dense tensor when needed
        elif isinstance(g, tf.IndexedSlices):
            g = tf.convert_to_tensor(g)
        processed_grads.append(g)

    # Global norm gradient clipping
    clipped_grads, _ = tf.clip_by_global_norm(processed_grads, clip_norm)

    # Return:
    # - current_state: state after applying all actions
    # - are_prec_sat: last precondition satisfaction flag
    # - goals_valid: result of goal checking on the final state
    # - total_loss: final scalar loss
    # - clipped_grads: list of clipped gradients aligned with variables_values
    return current_state, are_prec_sat, goals_valid, total_loss, clipped_grads


@tf.function
def execute_step_orig(initial_state, plan, variables_values):
    print("Execute gradient step")
    #return initial_state, 1, initial_state, tf.constant(0.0), variables_values[0], variables_values[1]
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(variables_values[0])
      tape.watch(variables_values[1])
      state=initial_state
      #loss, are_prec_sat, prec_satisfied, goals_satisfied, new_state = plan.forward(initial_state, variables_values)
      loss1, are_prec_sat, state = plan.forward_step(initial_state, variables_values[0])
      loss2, are_prec_sat, new_state = plan.forward_step(state, variables_values[1])

      goals_valid,metric_value_add=plan.check_goals(new_state, variables_values[1])
      if goals_valid<=0:
        state_values = tf.tensor_scatter_nd_add(new_state, indices=[[GlobalData.pos_metric_expr]], updates=[metric_value_add])
        loss=tf.gather(state_values,GlobalData.pos_metric_expr)      
      else:
        loss = loss2  # or however you want to aggregate it
    grad1 = tape.gradient(loss, variables_values[0])
    grad2 = tape.gradient(loss, variables_values[1])
    del tape
    return state,are_prec_sat,new_state,loss,grad1,grad2



#@tf.function
def plan_sequence1(initial_state, plan, learning_rate=0.1, steps=100):
  seq_plan = plan  # assuming `plan` is actually a TensorPlan object
  initial_state = change_initial_state(seq_plan, initial_state)

  variables_values = seq_plan.generate_variables_values()
  var_pos = seq_plan.tensor_state.get_key_position("large_container")
  var = tf.gather(initial_state, var_pos)

  tf.print("===================================================")
  tf.print("Orig Vars:", variables_values)
  variables_values_copy = tf.Variable(variables_values.numpy(), dtype=variables_values.dtype, trainable=True)
  prec_satisfied=1
  goals_satisfied=1
  for step in range(steps):
    with tf.GradientTape(persistent=False) as tape:
      tape.watch(variables_values)
      state=initial_state
      #loss, are_prec_sat, prec_satisfied, goals_satisfied, new_state = plan.forward(initial_state, variables_values)
      loss, are_prec_sat, state = plan.forward_step(initial_state, variables_values)
      loss, are_prec_sat, new_state = plan.forward_step(state, variables_values)
    
    grad = tape.gradient(loss, variables_values)

    # Convert IndexedSlices to dense if necessary
    if isinstance(grad, tf.IndexedSlices):
      grad = tf.convert_to_tensor(grad)

    # Clip gradients by global norm (recommended)
    clipped_grad, _ = tf.clip_by_global_norm([grad], clip_norm=10.0)
    clipped_grad = clipped_grad[0]  # unpack list

    if DEBUG> 0:
      variables_values_copy = tf.Variable(variables_values.numpy(), dtype=variables_values.dtype, trainable=True)

    # Gradient descent update
    variables_values.assign_sub(learning_rate * clipped_grad)

    if True or step % 10 == 0 or step == steps - 1:
      tf.print(f"Step {step} - Loss: {loss.numpy():.4f}")
      tf.print(step, " - Gradient:", grad, ", clipped:", clipped_grad)
      tf.print("Prec sat:", are_prec_sat)
      #tf.print("Vars:", variables_values)

      for i in range(len(GlobalData._class_variables_list)):
        tf.print("Variable", i, " name: ", GlobalData._class_variables_list[i], ", current: ", variables_values_copy[i], ", new value: ", variables_values[i])

      for i in range(seq_plan.tensor_state.size()):
        if state[i] != new_state[i]:
          tf.print("key1: ",seq_plan.tensor_state.get_key(i), "=", state[i])
        tf.print("key2: ",seq_plan.tensor_state.get_key(i), "=", new_state[i])
      tf.print("===================================================")

      tf.print("Prec sat:", prec_satisfied)
      tf.print("Goals sat:", goals_satisfied)
      os.sync()
  return loss


#@tf.function
def plan_sequenceA(initial_state, plan, learning_rate=0.1, steps=100):
    seq_plan = plan  # assuming `plan` is actually a TensorPlan object
    initial_state = change_initial_state(seq_plan, initial_state)

    # Get the variables to optimize
    variables_values = seq_plan.generate_variables_values()


    tf.print("===================================================")
    tf.print("===================================================")
    tf.print("Orig Vars:", variables_values)


    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(variables_values):
        with tf.GradientTape() as tape:
            loss, are_prec_sat, prec_sat, goals = plan.forward_step(initial_state, variables_values)
        grads = tape.gradient(loss, [variables_values])
        optimizer.apply_gradients(zip(grads, [variables_values]))
        return loss, are_prec_sat, grads

    for step in range(steps):
        loss, are_prec_sat, grads = train_step(variables_values)

        if step % 1 == 0 or step == steps - 1:
            tf.print(f"Step {step} - Loss: {loss.numpy():.4f}")
            tf.print("Gradient:", grads)
            tf.print("Prec sat:", are_prec_sat)
            tf.print("Vars:", variables_values)
            tf.print("===================================================")
    return loss

tensor_state.set_attr(large_container.name, 70)

#init_state=tensor_state.convert_to_Tf()
#print("Tensor init", tensor_state)

initial_state={}
initial_state["large_container"]=tf.constant(70.0)

tf.print("state", initial_state["large_container"] )
# Measure execution time of act_sequence
start_time = time.time()
#state=plan_sequence(initial_state,result.plan)
seq_plan=TfPlan(problem, tensor_state, sol_plan)
end_time = time.time()

tf.print("Creation time of act_sequence:", end_time - start_time, "seconds")
tf.print()

start_time = time.time()
#tf.print("Actions", act_list)
state_values=seq_plan.tensor_state.get_initial_state_values()
variables_values=seq_plan.generate_variables_values_sequential()

seq_plan.forward(state_values,variables_values[0])
state=seq_plan.get_state_values() 

end_time = time.time()
#tf.print("check state", check_state["large_container"] )
tf.print("1.Execution time of act_sequence:", end_time - start_time, "seconds")

tf.print()

#tensor_state.set_attr(large_container.name, 70)
#init_state=tensor_state.convert_to_Tf()
initial_state={}
initial_state["large_container"]=tf.constant(20.1)

tf.print("state2", initial_state["large_container"] )
#tf.print("state2", tensor_state["large_container"] )
# Measure execution time of act_sequence
start_time = time.time()
plan_sequence(initial_state, seq_plan)
state2=seq_plan.get_state_values() #.convert_to_Tf()
#seq_plan=TensorPlan(problem, result.plan)
#state=seq_plan.forward(initial_state)
end_time = time.time()

#tf.print("new state2", state2["large_container"] )

tf.print("2.Execution time of act_sequence:", end_time - start_time, "seconds")
#if state2==check_state:
#  tf.print("Equal")
#else:
#  tf.print("Not Equal")

tf.print()
#exit()

#tensor_state.set_attr(large_container.name, 40)
#init_state=tensor_state.convert_to_Tf()
#tf.print("state3", tensor_state["large_container"] )

#tf.print("Tensor init", tensor_state)
# Measure execution time of act_sequence


initial_state["large_container"]=tf.constant(20.1)
tf.print("state3", initial_state["large_container"] )
start_time = time.time()

#graphviz = GraphvizOutput()
#graphviz.output_file = 'basic.png'

#with PyCallGraph(output=graphviz):
  
plan_sequence(initial_state,seq_plan)


state3=seq_plan.get_state_values() #.convert_to_Tf()
#seq_plan=TensorPlan(problem, result.plan)
#state=seq_plan.forward(initial_state)
end_time = time.time()

#tf.print("new state3 cost ", state3["cost"] )
#tf.print("Actions", act_list)

tf.print("3.Execution time of act_sequence:", end_time - start_time, "seconds")
#if state3==check_state:
#  tf.print("Equal")
#else:
#  tf.print("Not Equal")

tf.print()


initial_state["large_container"]=tf.constant(20.0)
tf.print("state3a", initial_state["large_container"] )
start_time = time.time()
plan_sequence(initial_state,seq_plan)
state3=seq_plan.get_state_values() #.convert_to_Tf()
#seq_plan=TensorPlan(problem, result.plan)
#state=seq_plan.forward(initial_state)
end_time = time.time()

#tf.print("new state3", state3["large_container"] )
#tf.print("Actions", act_list)

#tf.print("new state3a cost  ", state3["cost"] )
tf.print("3a.Execution time of act_sequence:", end_time - start_time, "seconds")
#if state3==check_state:
#  tf.print("Equal")
#else:
#  tf.print("Not Equal")

tf.print()




for i in range(80, -10, -20):
  initial_state["large_container"]=tf.constant(i+0.1,dtype=tf.float32)
  tf.print("state", initial_state["large_container"] )
  start_time = time.time()
  plan_sequence(initial_state,seq_plan)
  state3=seq_plan.get_state_values() #.convert_to_Tf()
  #seq_plan=TensorPlan(problem, result.plan)
  #state=seq_plan.forward(initial_state)
  end_time = time.time()

  #tf.print("new state4 cost ", state3["cost"] )
  #tf.print("Actions", act_list)

  tf.print(i,".Execution time of act_sequence:", end_time - start_time, "seconds")

  #if state3==check_state:
  #  tf.print("Equal")
  #else:
  #  tf.print("Not Equal")

  tf.print()

exit()

# Write the graph to TensorBoard logs
with writer.as_default():
    tf.summary.graph(act_sequence.get_concrete_function(initial_state,sol_plan).graph)

#apply action


    # Use the `effect.value` directly if it's a constant, otherwise evaluate
    #  if effect.value.is_constanexpressiont():
    #    new_state[effect.fluent] = effect.value.constant_value()
    #  else:
    #    raise ValueError("Only constant effects are supported in this example")
 
    #if (simulator.is_applicable(new_state, fill_large)):
    #  tf.print("Action applicable")
    #else:
    #  tf.print("Action not applicable")

    #new_state = simulator.apply(new_state, fill_large)
    #tf.print("new state: ",new_state)   

    #if (simulator.is_applicable(new_state, fill_large)):
    #  tf.print("Action applicable")
    #else:
    #  tf.print("Action not applicable")

    #new_state = simulator.apply(new_state, fill_large)
    #tf.print("new state: ",new_state)   

    #init = simulator.get_state_values()
    #print("init", init)

    #c1=myadd(a1,b1)
    
    #d=myadd(b1,c1)
    #return d

#Parsing 

#x = tf.Variable(1.0, dtype=tf.float32)
#y = tf.Variable(2.0, dtype=tf.float32)
#z = tf.Variable(3.0, dtype=tf.float32)


#variables = {'x': x, 'y': y, 'z': z}
#result = sympy_to_tensorflow(sympy_expr, variables)
 


# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/fit

import math
def  mypow(c):
  return c*c
def myadd(a, b):
  b= mypow(b)+a
  b= mypow(b)
  print(b)
  return a + b
@tf.function  # The decorator converts `add` into a `PolymorphicFunction`.
def add(a, b):
  return (myadd(a,b))


#a = tf.Variable(1.0)
#b=tf.constant(2.0)
# Write the graph to TensorBoard logs
#with writer.as_default():
#    tf.summary.graph(add.get_concrete_function(a,b ).graph)



#simulator=SequentialSimulator(problem)
#init = simulator.get_initial_state()
#print("init", init)

# Simulate executing the "fill_large" action
state_after_action = apply_action(problem, initial_state, fill_large)

# Print the updated state after attempting to execute the action
#print("State after attempting fill_large action:", {fluent: state_after_action[fluent] for fluent in state_after_action})


# Define a log directory with a timestamp to avoid overwrites
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print("log_dir:", log_dir)
writer = tf.summary.create_file_writer(log_dir)

with writer.as_default():
    tf.summary.graph(apply_action.get_concrete_function(problem,  initial_state, fill_large).graph)

##################################################


# Initialize the planning problem (as previously set up)
problem = Problem("WaterTransferProblem")

# Define the fluent state variables
large_container = Fluent("large_container", IntType())
small_container = Fluent("small_container", IntType())

# Define actions
fill_large = InstantaneousAction("fill_large")

fill_large.add_precondition(Equals(large_container, 1))
fill_large.add_effect(large_container, 10)

# Add fluents and actions to the problem
problem.add_fluent(large_container)
problem.add_fluent(small_container)
problem.add_action(fill_large)

# Set the initial state
initial_state = {large_container.name: 20, small_container.name: 0}
problem.set_initial_value(large_container, 110)
problem.set_initial_value(small_container, 0)


# Define a helper function to apply an action's effect
def apply_action(state, action):
    # Create a copy of the state to apply effects
    new_state = state.copy()
    for effect in action.effects:
        # Use the `effect.value` directly if it's a constant, otherwise evaluate
        if effect.value.is_constant():
            new_state[effect.fluent] = effect.value.constant_value()
        else:
            raise ValueError("Only constant effects are supported in this example")

    # Print the updated state after executing the action
    print("State after executing fill_large action:",
      {fluent: new_state[fluent] for fluent in new_state})
    return new_state

# Simulate executing the "fill_large" action
state_after_action = apply_action(initial_state, fill_large)

# Print the updated state after executing the action
print("State after executing fill_large action:",
      {fluent: state_after_action[fluent] for fluent in state_after_action})


#####################################################

# Commented out IPython magic to ensure Python compatibility.
from pickle import NONE
from unified_planning.shortcuts import *

from unified_planning.engines import UPSequentialSimulator, SequentialSimulatorMixin
from unified_planning.model import State
from unified_planning.plans import ActionInstance
from unified_planning.test import unittest_TestCase, main
from unified_planning.test.examples import get_example_problems
from unified_planning.exceptions import UPUsageError

@tf.function
def myPlan(x):
  counter = Fluent("counter", IntType())
  increase = InstantaneousAction("increase")
  increase.add_increase_effect(counter, 1)
  decrease = InstantaneousAction("decrease")

  decrease.add_precondition(GE(counter, 3))
  decrease.add_decrease_effect(counter, 1)
  problem = Problem("simple_counter")
  problem.add_fluent(counter, default_initial_value=5)
  problem.add_action(increase)
  problem.add_action(decrease)
  #print("problem:", problem)

  simulator=SequentialSimulator(problem)
  state=init = simulator.get_initial_state()

  #assertTrue(simulator.is_applicable(init, increase))
  print("Initial state:", init)  # Print initial state


  dec_state = simulator.apply(init, decrease)
  #assert dec_state is not NONE

  print("dec state:", dec_state)
  print(simulator.is_applicable(dec_state, decrease))

  state=double_dec_state = simulator.apply(dec_state, decrease)
  #assertIsNone(double_dec_state)
  #assert double_dec_state is not None
  print("dec state:", double_dec_state)

  return 0 #double_dec_state

with writer.as_default():
    tf.summary.graph(myPlan.get_concrete_function(1).graph)


#print("out:", out)

"""Now we start to model a problem involving three numeric variables $c_0$, $c_1$ and $c_2$ that can be increased and decreased. The goal of this problem is to change the variables values such that  $c_0 < c_1 < c_2$. We name with value the lifted fluent that lets us access to the value of a given counter $c$.



"""

x = tf.Variable([3.0])
#x=[3.0]

problems={}
# basic numeric
value = Fluent("value", IntType())
task = InstantaneousAction("task")
task.add_precondition(Equals(value, 1))
task.add_effect(value, 2)
problem = Problem("basic_numeric")
problem.add_fluent(value)
problem.add_action(task)
problem.set_initial_value(value, 0)
problem.add_goal(Equals(value, 2))
plan = up.plans.SequentialPlan([up.plans.ActionInstance(task)])
problems["basic_numeric"] = TestCase(
  problem=problem, solvable=True, valid_plans=[plan]
)

"""
#### Creating the fluent

First, we define the `UserTypes` and the `Fluents`."""

Counter = UserType('Counter')

value = Fluent('value', RealType(), m=Counter)

print(value)

print(Counter)

"""#### Creating the actions

"""

inc = InstantaneousAction('increment',c=Counter)
c = inc.parameter('c')
inc.add_precondition(LE(value(c), 10))
inc.add_increase_effect(value(c), 1)

dec = InstantaneousAction('decrement',c=Counter)
c = dec.parameter('c')
dec.add_precondition(GT(value(c), 0))
dec.add_decrease_effect(value(c),1)

print(dec)

"""Finally, we can create a `Problem` that encompasses the fluents and the actions, and puts them together with concrete objects, an initial state and a goal. Note here that we do not need to specify all values for each object. These are set to 0 using the default intial value parameter.

"""

problem = Problem('problem')

problem.add_fluent(value, default_initial_value=0)
C0 = Object('c0', Counter)
C1 = Object('c1', Counter)
C2 = Object('c2', Counter)
problem.add_object(C0)
problem.add_object(C1)
problem.add_object(C2)
problem.add_action(inc)
problem.add_action(dec)
problem.add_goal(And( GE(value(C2),Plus(value(C1),1)), GE(value(C1),Plus(value(C0),1))))
problem

"""
Now we see how we can generate another, larger problem, much more compactly using a more programmatic definition

"""

N = 9 # This is the number of counters

p2 = Problem('Large_problems')

p2.add_fluent(value, default_initial_value=0)
p2.add_objects([Object(f'c{i}',Counter) for i in range(N)])
p2.add_action(inc)
p2.add_action(dec)

for i in range(N-1):
    p2.add_goal(GE(value(p2.object(f'c{i+1}')),Plus(value(p2.object(f'c{i}')),1)))

p2

"""#### Solving the small and the parametric problem

The unified_planning can either select among the available planners one which is suited for the task at hand (looking at the problem kind), or use the user defined planning. In what follows we first attempt to solve the small problem with three counters and ask the UP to use a specific planning system (ENHSP), and then one with N=9 counters (problem p2) asking the UP to automatically select the engine

"""

with OneshotPlanner(name='enhsp') as planner:
    result = planner.solve(problem)
    plan = result.plan
    if plan is not None:
        print("%s returned:" % planner.name)
        print(plan)
    else:
        print("No plan found.")

with OneshotPlanner(problem_kind=problem.kind) as planner:
    result = planner.solve(p2)
    plan = result.plan
    if plan is not None:
        print("%s returned:" % planner.name)
        print(plan)
    else:
        print("No plan found.")

"""Now let us create a problem medium-sized problem, set up a minimisation function as minimize the number of actions, and see how this can be solved optimally."""

from unified_planning.model.metrics import MinimizeSequentialPlanLength

N = 7 #This is the number of counters

mediumSizeProblem = Problem('Medium_sized_problem')

mediumSizeProblem.add_fluent(value, default_initial_value=0)
mediumSizeProblem.add_objects([Object(f'c{i}',Counter) for i in range(N)])
mediumSizeProblem.add_action(inc)
mediumSizeProblem.add_action(dec)
metric = MinimizeSequentialPlanLength()
mediumSizeProblem.add_quality_metric(metric)

for i in range(N-1):
    mediumSizeProblem.add_goal(GE(value(p2.object(f'c{i+1}')),Plus(value(p2.object(f'c{i}')),1)))

with OneshotPlanner(problem_kind=problem.kind,optimality_guarantee=True) as planner:
    result = planner.solve(mediumSizeProblem)
    plan = result.plan
    if plan is not None:
        print("%s returned:" % planner.name)
        print(plan)
    else:
        print("No plan found.")
