
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


#Profiling

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


"""UTILS"""
EPSILON=1e-7  #tf.keras.backend.epsilon()
SOL_FILE="result_plan.sol"


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
    url = tb.launch()
    print(f"Tensorflow listening on {url}")



# Planning Demo
############################################################
# Initialize the planning problem
problem = Problem("WaterTransferProblem")

# Define fluent state variables
large_container = Fluent("large_container", IntType())
small_container = Fluent("small_container", IntType())

cost = Fluent("cost", IntType())

# Define actions
fill_large = InstantaneousAction("fill_large")
#fill_large.add_effect(large_container,Plus(large_container, small_container))
fill_large.add_increase_effect(cost,Times(large_container, small_container))
fill_large.add_decrease_effect(small_container,1)

fill_large.add_increase_effect(large_container, small_container)
# Example precondition: Only fill if large_container is empty
fill_large.add_precondition(LT(large_container, 80))
fill_large.add_precondition(GE(large_container, 0))
fill_large.add_precondition(LT(Times(large_container, small_container), 800))

# Add conditional effect
fill_large.add_effect( cost, Plus(cost, large_container), GT(small_container, 10)) 

# Add fluents and actions to the problem
problem.add_fluent(large_container)
problem.add_fluent(small_container)
problem.add_fluent(cost)
problem.add_action(fill_large)
#problem.set_initial_value(large_container, tf.constant(20))
problem.set_initial_value(large_container, 70)
problem.set_initial_value(small_container, 5)

problem.set_initial_value(cost, 0)

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

move.add_effect(robot_at(l_from), False)
move.add_effect(robot_at(l_to), True)
move.add_decrease_effect(large_container,5)
move.add_increase_effect(cost, large_container)

print(move)
problem.add_fluent(robot_at, default_initial_value=False)
problem.add_fluent(connected, default_initial_value=False)
problem.add_action(move)

NLOC = 10
locations = [unified_planning.model.Object('l%s' % i, Location) for i in range(NLOC)]
problem.add_objects(locations)

problem.set_initial_value(robot_at(locations[0]), True)
for i in range(NLOC - 1):
    problem.set_initial_value(connected(locations[i], locations[i+1]), True)

problem.add_goal(robot_at(locations[-1]))
problem.add_goal(GE(large_container, 10))

print("Prolem: ",problem)

metric = MinimizeSequentialPlanLength()
problem.add_quality_metric(metric)


# Solve the planning problem
sol_plan=None
if not os.path.exists(SOL_FILE):
  with OneshotPlanner(name='pyperplan') as planner:
    result = planner.solve(problem)
    if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
        print("Pyperplan returned: %s" % result.plan)
        # Save result.plan to a file
        writer = PDDLWriter(problem)
        writer.write_plan(result.plan, "result_plan.sol")
    else:
        print("No plan found.")


# Set the initial state
#initial_state = {large_container:  tf.constant(1.0), small_container:  tf.constant(0, dtype=tf.float32)}
initial_state ={}

tensor_state=TfState(problem)
if os.path.exists(SOL_FILE):
  # Reload the saved plan from the file
  # Reload the saved PDDL solution file
  reader = PDDLReader()
  sol_plan = reader.parse_plan(problem,"result_plan.sol")
  
@tf.function
def plan_sequence(my_state, plan):
  seq_plan=plan #TensorPlan(problem, plan)

  var=seq_plan.states_sequence[0]["large_container"]
  with tf.GradientTape() as tape:
    tape.watch(seq_plan.states_sequence[0]['large_container'])
    loss=plan.forward(initial_state)

  grad = tape.gradient(loss, var)
  tf.print("Grad: ",grad)
  state=seq_plan.get_state()
  #tf.print("new state cost: ", state["cost"] )
  #value=seq_plan.get_plan_metric()
  #tf.print("Plan metric:", value)
  return {} #state.convert_to_Tf()


tensor_state.set_attr(large_container.name, 70)

#init_state=tensor_state.convert_to_Tf()
#print("Tensor init", tensor_state)

initial_state={}
initial_state["large_container"]=tf.constant(70.0)

print("state", initial_state["large_container"] )
# Measure execution time of act_sequence
start_time = time.time()
#state=plan_sequence(initial_state,result.plan)
seq_plan=TfPlan(problem, tensor_state, sol_plan)
end_time = time.time()
print("Creation time of act_sequence:", end_time - start_time, "seconds")
print()

start_time = time.time()
#print("Actions", act_list)
seq_plan.forward(initial_state)
state=seq_plan.get_state() 

end_time = time.time()
#print("check state", check_state["large_container"] )
print("1.Execution time of act_sequence:", end_time - start_time, "seconds")

print()

#tensor_state.set_attr(large_container.name, 70)
#init_state=tensor_state.convert_to_Tf()
initial_state={}
initial_state["large_container"]=tf.constant(20.1)

print("state2", initial_state["large_container"] )
#print("state2", tensor_state["large_container"] )
# Measure execution time of act_sequence
start_time = time.time()
plan_sequence(initial_state, seq_plan)
state2=seq_plan.get_state() #.convert_to_Tf()
#seq_plan=TensorPlan(problem, result.plan)
#state=seq_plan.forward(initial_state)
end_time = time.time()

#print("new state2", state2["large_container"] )

print("2.Execution time of act_sequence:", end_time - start_time, "seconds")
#if state2==check_state:
#  print("Equal")
#else:
#  print("Not Equal")

print()

var=seq_plan.states_sequence[0]["large_container"]

start_time = time.time()

end_time = time.time()
state=seq_plan.get_final_state() 
with tf.GradientTape() as tape:
  tape.watch(seq_plan.states_sequence[0]['large_container'])
  loss=seq_plan.forward(initial_state)

grad = tape.gradient(loss, var)
#print("Final state:", seq_plan.get_final_state())
print("Grad: ",grad)

print("new state cost", state["cost"] )
print("Execution time of act_sequence:", end_time - start_time, "seconds")
print()



#tensor_state.set_attr(large_container.name, 40)
#init_state=tensor_state.convert_to_Tf()
#print("state3", tensor_state["large_container"] )

#print("Tensor init", tensor_state)
# Measure execution time of act_sequence


initial_state["large_container"]=tf.constant(20.1)
print("state3", initial_state["large_container"] )
start_time = time.time()

#graphviz = GraphvizOutput()
#graphviz.output_file = 'basic.png'

#with PyCallGraph(output=graphviz):
  
plan_sequence(initial_state,seq_plan)


state3=seq_plan.get_state() #.convert_to_Tf()
#seq_plan=TensorPlan(problem, result.plan)
#state=seq_plan.forward(initial_state)
end_time = time.time()

print("new state3 cost ", state3["cost"] )
#print("Actions", act_list)

print("3.Execution time of act_sequence:", end_time - start_time, "seconds")
#if state3==check_state:
#  print("Equal")
#else:
#  print("Not Equal")

print()


initial_state["large_container"]=tf.constant(20.0)
print("state3a", initial_state["large_container"] )
start_time = time.time()
plan_sequence(initial_state,seq_plan)
state3=seq_plan.get_state() #.convert_to_Tf()
#seq_plan=TensorPlan(problem, result.plan)
#state=seq_plan.forward(initial_state)
end_time = time.time()

#print("new state3", state3["large_container"] )
#print("Actions", act_list)

print("new state3a cost  ", state3["cost"] )
print("3a.Execution time of act_sequence:", end_time - start_time, "seconds")
#if state3==check_state:
#  print("Equal")
#else:
#  print("Not Equal")

print()




for i in range(80, -10, -5):
  initial_state["large_container"]=tf.constant(i+0.1,dtype=tf.float32)
  print("state", initial_state["large_container"] )
  start_time = time.time()
  plan_sequence(initial_state,seq_plan)
  state3=seq_plan.get_state() #.convert_to_Tf()
  #seq_plan=TensorPlan(problem, result.plan)
  #state=seq_plan.forward(initial_state)
  end_time = time.time()

  print("new state4 cost ", state3["cost"] )
  #print("Actions", act_list)

  print("4.Execution time of act_sequence:", end_time - start_time, "seconds")

  #if state3==check_state:
  #  print("Equal")
  #else:
  #  print("Not Equal")

  print()


tensor_state.set_attr(large_container.name, 40)
print("state4", tensor_state["large_container"] )
# Measure execution time of act_sequence
start_time = time.time()
#state=plan_sequence(initial_state,result.plan)
seq_plan=TfPlan(problem, sol_plan)
seq_plan.forward(tensor_state)
state4=seq_plan.get_state() #.convert_to_Tf()
value=seq_plan.get_plan_metric()
print("Plan metric:", value)
end_time = time.time()
#print("new state4", state4["large_container"] )
#print("Actions", act_list)

print("4.Execution time of act_sequence:", end_time - start_time, "seconds")
if state4==check_state:
  print("Equal")
else:
  print("Not Equal")


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
    #  print("Action applicable")
    #else:
    #  print("Action not applicable")

    #new_state = simulator.apply(new_state, fill_large)
    #print("new state: ",new_state)   

    #if (simulator.is_applicable(new_state, fill_large)):
    #  print("Action applicable")
    #else:
    #  print("Action not applicable")

    #new_state = simulator.apply(new_state, fill_large)
    #print("new state: ",new_state)   

    #init = simulator.get_state()
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
