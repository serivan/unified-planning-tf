#!/usr/bin/env python
# coding: utf-8

"""We are now ready to use the Unified-Planning library!

but first import libraries for pytorch
"""

# Commented out IPython magic to ensure Python compatibility.
## Standard libraries
import os, shutil
import math
import numpy as np
import time

os.environ["TF_AUTOGRAPH_CACHE_DIR"] = "/mnt/ramdisk/tensorflow"
os.makedirs(os.environ["TF_AUTOGRAPH_CACHE_DIR"] , exist_ok=True)

import os

# Set the TMPDIR environment variable to your ramdisk directory
os.environ["TMPDIR"] = "/mnt/ramdisk/tensorflow"

# Ensure the directory exists
os.makedirs("/mnt/ramdisk/tf_temp", exist_ok=True)

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
#int = lambda value: tf.constant(value, dtype=tf.Tensor)
import tensorboard
from tensorboard import main as tb
from tensorboard import default
from tensorboard import program

import traceback
import contextlib



import sympy as sp

import datetime
import pickle
import random
#import psutil
import tracemalloc

tracemalloc.start()


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
"""UTILS"""

#tf.config.optimizer.set_experimental_options({"autotune": True})
#tf.config.run_functions_eagerly(True)


# create the new directory for the pickle files
new_folder = os.path.join(os.getcwd(), 'pickle_files/')
os.makedirs(new_folder, exist_ok=True)

# PDDL problem from the file using pickle
#with open(os.path.join(new_folder,'pddl_problem_2001.pkl'), 'rb') as file:
#  w_problem = pickle.load(file)

def get_memory():
  current, peak = tracemalloc.get_traced_memory()
  print(f"Memoria attuale: {current / 1024**3:.6f} GB")
  print(f"Memoria massima usata: {peak / 1024**3:.6f} GB")
  load_avg = os.getloadavg()  # Restituisce il carico medio su 1, 5 e 15 minuti
  print(f"Load Average (1, 5, 15 min): {load_avg}")
   
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

GlobalData._class_tensor_state=tensor_state

#TensorState.print_filtered_dict_state(tensor_state.get_tensors(),["next"])
#exit()
#print("Tensor state: ",tensor_state)
end_time = time.time()
print("State creation:", end_time - start_time, "seconds")
get_memory()
print()
os.sync()

def change_initial_state(plan, initial_state):
  state_values=plan.tensor_state.get_initial_state_values()
  for fluent, value in initial_state.items():
        #print("Fluent:", fluent)
        #print("Initial value:", value)

        pos=plan.tensor_state.get_key_position(fluent)
        if pos>=0:
            state_values[pos].assign(value)
            tf.print("Fluent: ", fluent, " value: ", state_values[pos])

  
  storage=plan.tensor_state.get_key_position("storage(hoa_binh_dam)") 
  #tf.print("Storage pos:", storage, " value: ", state_values[storage])
  
  return state_values
iter= tf.Variable(0, dtype=tf.int32) 
@tf.function #(reduce_retracing=True) #(experimental_relax_shapes=True)  # #(jit_compile=True)
def execute(plan, initial_state_values):
  iter.assign_add(1)
  if DEBUG>=0:
    tf.print("Execute iter: ", iter)
  result= plan.forward(initial_state_values)
  return result

initial_state={}
start_time = time.time()
#state=plan_sequence(initial_state,result.plan)
seq_plan=TfPlan(w_problem, tensor_state, sol_plan)
end_time = time.time()
print("Creation time of act_sequence:", end_time - start_time, "seconds")
get_memory()
os.sync()
#DEBUG=6
print()
#exit()

initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(370.0)
print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
start_time = time.time()
result= 0
#result= seq_plan.preprocess_apply(initial_state)
end_time = time.time()
print("Preprocess:", end_time - start_time, "seconds, result: ", result)
get_memory()




print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
initial_state_values=change_initial_state(seq_plan, initial_state)
start_time = time.time()
result= seq_plan.forward(initial_state_values)
#result= 0
end_time = time.time()
print("Forward1A:", end_time - start_time, "seconds, result: ", result)
get_memory()

os.sync()
#exit()

os.sync()
#exit()
time.sleep(5)  # Pauses execution for 5 seconds
print("START")
tf.print("TF START")


initial_state={}
initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(380.0)
print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
initial_state_values=change_initial_state(seq_plan, initial_state)
start_time = time.time()
result= seq_plan.forward(initial_state_values)
#result= 0
end_time = time.time()
print("Forward1B:", end_time - start_time, "seconds, result: ", result)
get_memory()

#

initial_state={}
initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(380.0)
print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
initial_state_values=change_initial_state(seq_plan, initial_state)
start_time = time.time()
result= seq_plan.forward(initial_state_values)
#result= 0
end_time = time.time()
print("Forward1C:", end_time - start_time, "seconds, result: ", result)
get_memory()
#exit()

# Start profiling

# Start TensorFlow Profiler
TBOARD= True #False
use_callgraph = False
use_cProfile = False



initial_state={}
initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(3700.0)
initial_state_values=change_initial_state(seq_plan, initial_state)
result= execute(seq_plan, initial_state_values)

initial_state_values=change_initial_state(seq_plan, initial_state)
if TBOARD:
  logdir = "/tmp/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tf.profiler.experimental.start(logdir)
  print("TBoard Logdir: ", logdir)
  with tf.profiler.experimental.Trace('execute', step_num=1, _r=1):
      result= execute(seq_plan, initial_state_values)
  #
  tf.profiler.experimental.stop()

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

    result= execute(seq_plan, initial_state_values)

if use_cProfile:
  import cProfile
  cProfile.run('result= execute(seq_plan, initial_state_values)')

end_time = time.time()
print("Forward2:", end_time - start_time, "seconds, result: ", result)
print()
#exit()

initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(370.0)
print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
initial_state_values=change_initial_state(seq_plan, initial_state)
start_time = time.time()
result= execute(seq_plan, initial_state_values)
end_time = time.time()
print("Forward3:", end_time - start_time, "seconds, result: ", result)
get_memory()
print("\n====================================")
times=[]
for i in range(0,100):
  val=random.randint(0,1500)
  initial_state["agricultural_demand(day_2001_10_01)"]=tf.constant(val, dtype=tf.float32)
  print("set initial state: ",initial_state["agricultural_demand(day_2001_10_01)"])
  initial_state_values=change_initial_state(seq_plan, initial_state)
  
  start_time = time.time()
  result= execute(seq_plan, initial_state_values)
  end_time = time.time()
  delta=end_time - start_time
  print("Forward-"+str(i)+": ", end_time - start_time, "seconds, result: ", result)
  get_memory()
  times.append(delta)
  print()

print("Average time: ", np.mean(times), "seconds")
print("Standard deviation: ", np.std(times), "seconds")

os.sync()

# %%
