#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

import datetime
import pickle
import random

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

from unified_planning.io import PDDLWriter, PDDLReader

from unified_planning.tensor.constants import *

from tensorflow.python.eager import profiler
tf.config.optimizer.set_experimental_options({"autotune": True})
"""UTILS"""

# create the new directory for the pickle files
new_folder = os.path.join(os.getcwd(), 'pickle_files/')
os.makedirs(new_folder, exist_ok=True)

# PDDL problem from the file using pickle
#with open(os.path.join(new_folder,'pddl_problem_2001.pkl'), 'rb') as file:
#  w_problem = pickle.load(file)


reader = PDDLReader()

#SOL_FILE="plan.sol"
SOL_FILE="plan2.sol"
#SOL_FILE="SProblem_2001-09-30_end_2002-05-31.plan"

#w_problem = reader.parse_problem(new_folder + 'domain.pddl', new_folder + 'problem.pddl')
w_problem = reader.parse_problem(new_folder + 'domain.pddl', new_folder + 'problem2.pddl')
#w_problem = reader.parse_problem(new_folder + 'domain.pddl', new_folder + 'SProblem_2001-09-30_end_2002-05-31.pddl')

if os.path.exists(new_folder+SOL_FILE):
  # Reload the saved plan from the file
  # Reload the saved PDDL solution file
  sol_plan = reader.parse_plan(w_problem,new_folder+SOL_FILE)
if(DEBUG>=0):   
  print("Actions:")
  print(w_problem.actions)

#writer = PDDLWriter(problem=w_problem)
#writer.write_problem(new_folder + 'problem.pddl')
#writer.write_domain( new_folder + 'domain.pddl')

start_time = time.time()
tensor_state=TfState(w_problem)


#TensorState.print_filtered_dict_state(tensor_state.get_tensors(),["next"])
#exit()
#print("Tensor state: ",tensor_state)
end_time = time.time()
print("State creation:", end_time - start_time, "seconds")
print()
os.sync()

@tf.function(reduce_retracing=True)
def execute(plan, initial_state):
  result= plan.forward(initial_state)
  state=plan.get_final_state()
  tf.print("E. Objective function: ", state["objective"] )
  return result

initial_state={}
start_time = time.time()
#state=plan_sequence(initial_state,result.plan)
seq_plan=TfPlan(w_problem, tensor_state, sol_plan)
end_time = time.time()
print("Creation time of act_sequence:", end_time - start_time, "seconds")
#state=seq_plan.get_final_state()
#print("Creation Objective function: ", state["objective"] )
os.sync()
#DEBUG=6
print()
#exit()

initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(3700.0)
print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
start_time = time.time()
result= 0
#result= seq_plan.preprocess_apply(initial_state)
end_time = time.time()
print("Preprocess:", end_time - start_time, "seconds, result: ", result)
#state=seq_plan.get_final_state()
#print("Objective function: ", state["objective"] )


initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(3700.0)
print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
start_time = time.time()
result= seq_plan.forward(initial_state)
#result= 0
end_time = time.time()
print("Forward1A:", end_time - start_time, "seconds, result: ", result)
state=seq_plan.get_final_state()
print("Objective function: ", state["objective"] )
os.sync()
#exit()

initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(370.0)
print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
start_time = time.time()
result= seq_plan.forward(initial_state)
#result= 0
end_time = time.time()
print("Forward1B:", end_time - start_time, "seconds, result: ", result)
state=seq_plan.get_final_state()
print("Objective function: ", state["objective"] )

os.sync()
#exit()
time.sleep(5)  # Pauses execution for 5 seconds
print("START")
tf.print("TF START")


initial_state={}
initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(3800.0)
print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
start_time = time.time()

#
# Start profiling

# Start TensorFlow Profiler
logdir = "/tmp/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tf.profiler.experimental.start(logdir)

#with tf.profiler.experimental.Trace('execute', step_num=1, _r=1):
#    result= execute(seq_plan, initial_state)
#
tf.profiler.experimental.stop()



use_callgraph = False
if use_callgraph:
  #Profiling
  # Generate call graph
  graphviz = GraphvizOutput(output_file='callgraph.svg')  # Vector format
  graphviz.output_type = 'svg'  # High-resolution, scalable

  #graphviz = GraphvizOutput(output_file='callgraph.pdf')
  #graphviz.output_type = 'pdf' 
  #graphviz.dot_args = ['-Gdpi=300'] 

  from pycallgraph import PyCallGraph
  from pycallgraph.output import GraphvizOutput
  with PyCallGraph(output=graphviz):
    result= execute(seq_plan, initial_state)
else:
  import cProfile
  #cProfile.run('result= execute(seq_plan, initial_state)')
  result= execute(seq_plan, initial_state)
end_time = time.time()
print("Forward2:", end_time - start_time, "seconds, result: ", result)
print()


initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(370.0)
print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
start_time = time.time()
result= execute(seq_plan, initial_state)
end_time = time.time()
print("Forward3:", end_time - start_time, "seconds, result: ", result)
print()
times=[]
for i in range(0,100):
  val=random.randint(0,1500)
  initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(val, dtype=tf.float32)
  print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
  start_time = time.time()
  result= execute(seq_plan, initial_state)
  end_time = time.time()
  delta=end_time - start_time
  print("Forward-"+str(i)+": ", end_time - start_time, "seconds, result: ", result)
  #state=seq_plan.get_final_state()
  #print("Objective function: ", state["objective"] )
  times.append(delta)
  print()

print("Average time: ", np.mean(times), "seconds")
print("Standard deviation: ", np.std(times), "seconds")

os.sync()

# %%
