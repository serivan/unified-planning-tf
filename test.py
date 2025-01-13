import tensorflow as tf

import time
import random
import copy

import tensorflow as tf

# Dimensione dell'array
size = 5

# Creazione di un TensorArray
tensor_array = tf.TensorArray(dtype=tf.float32, size=size, dynamic_size=False, clear_after_read=False)

# Funzione per creare variabili con flag `trainable`
def create_variable(value, trainable):
    return tf.Variable(value, trainable=trainable, dtype=tf.float32)

# Inizializzazione delle variabili con trainable misto
variables = [
    create_variable(value=1.0, trainable=True),  # Trainable
    create_variable(value=2.0, trainable=True), # Non trainable
    create_variable(value=3.0, trainable=True),  # Trainable
    create_variable(value=4.0, trainable=False), # Non trainable
    create_variable(value=5.0, trainable=True),  # Trainable
]

# Scrivere le variabili nel TensorArray
for i, var in enumerate(variables):
    tensor_array = tensor_array.write(i, var)

# Leggere e stampare le variabili dal TensorArray
for i in range(tensor_array.size()):
    print(f"Variable at index {i}: {tensor_array.read(i).numpy()} (trainable={variables[i].trainable})")

# Define a random value generator
def rnd():
    return random.uniform(0, 10)

#@tf.function
def parse_and_build_graph(operation_string, inputs):
    """
    Parse a string defining a TensorFlow operation and build a reusable computational graph.

    Args:
        operation_string (str): A string defining the TensorFlow operations (e.g., "a + b", "tf.add(a, b)").
        inputs (dict): A dictionary mapping variable names to TensorFlow tensors.

    Returns:
        function: A TensorFlow function that can be executed multiple times.
    """
    # Define the graph-building function
    
    def graph_func():
        # Dynamically evaluate the operation string in the given scope
        # Add TensorFlow's namespace to the evaluation environment
        local_scope = {**inputs, "tf": tf} 
        var=(inputs['a'], inputs['b'])
        with tf.GradientTape() as tape:
            tape.watch(var)
            if( inputs['a'] > 2):
                loss=eval(operation_string, {}, local_scope)
            else:
                loss=eval(operation_string+"+ a", {}, local_scope)
        grad = tape.gradient(loss, var)
        tf.print("Var: ", var, " Grad: ",grad)
 
        return loss

    return graph_func()

M=5
# Example Usage
# Define inputs
a = tf.Variable(5.0)
b = tf.Variable(3.0)
inputs = {"a": a, "b": b}

inputs["c"]=tf.Variable(7.0)

# Parse a string and build the graph
operation_string = "tf.add(a, b) * 2"
graph = parse_and_build_graph #(operation_string, inputs)

# Execute the graph multiple times
for i in range(M):
    a.assign(rnd())
    start_time = time.time()
    result = parse_and_build_graph(operation_string, inputs)
    end_time = time.time()

    print(f"Execution {i+1}: result={result}, time={end_time - start_time:.6f} seconds")

print()
# Parse a string and build the graph
operation_string = "tf.multiply(a, b) * 3"
# Execute the graph multiple times
for i in range(M):
    a.assign(rnd())
    start_time = time.time()
    result = graph(operation_string, inputs).numpy()
    end_time = time.time()

    print(f"Execution {i+1}: result={result}, time={end_time - start_time:.6f} seconds")

print()
# Parse a string and build the graph
operation_string = "tf.add(a, b) * 2"
print("copy inputs")
inputs_copy = copy.deepcopy(inputs)  # Create a deep copy of the dictionary
# Execute the graph multiple times
for i in range(M):
    a.assign(rnd())
    start_time = time.time()
    result = graph(operation_string, inputs_copy).numpy()
    end_time = time.time()

    print(f"Execution {i+1}: result={result}, time={end_time - start_time:.6f} seconds")


print()
operation_string = "tf.reduce_sum(inputs)"

operation_string = "inputs[1] + inputs[2]"

@tf.function
def graph_func(operation_string, inputs):
    local_scope = {"tf": tf, "inputs": inputs}
    return eval(operation_string, {}, local_scope)

# Example usage
def generate_random_tensor(shape):
    return tf.constant([random.uniform(0, 10) for _ in range(shape[0])])

# Example usage
inputs2 = tf.constant([1.0, 2.0, 3.0, 4.0])
i=0
start_time = time.time()
result = graph_func(operation_string, inputs2)
end_time = time.time()
print(f"Execution {i}: result={result}, time={end_time - start_time:.6f} seconds")


print(f"Result: {result.numpy()}")  # Output: 10.0
# Execute the graph multiple times
for i in range(M):
    start_time = time.time()
    inputs2 = generate_random_tensor([4])  # Assign random values to inputs2
    result = graph_func(operation_string, inputs2)
    end_time = time.time()

    print(f"Execution {i+1}: result={result}, time={end_time - start_time:.6f} seconds")

