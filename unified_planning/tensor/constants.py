import tensorflow as tf
import functools
import inspect

# Activate debug mode
DEBUG = 0

# Define constants
EPSILON=1e-7  #tf.keras.backend.epsilon()
TENSOR_EPSILON = 1e-6
ARE_PREC_SATISF_STR='ARE_PREC_SATISFIED'

TF_ACTV_FN_BOOL = tf.nn.tanh
TF_ACTV_FN_REAL = tf.nn.relu
TF_ZERO=tf.Variable(0.0, dtype=tf.float32, trainable=False)

def grep( string, pattern):
    """
    Mimic the behavior of the grep command on a Python string.

    Args:
        pattern (str): The pattern to search for.
        string (str): The string to search within.

    Returns:
        str: The lines that match the pattern.
    """
    matching_lines = []
    for line in string.splitlines():
        if pattern in line:
            matching_lines.append(line)
    return "\n".join(matching_lines)



def get_closure_vars(lambda_func):
    """
    Extract captured variables from a lambda function's closure.

    Args:
        lambda_func (callable): The lambda function to inspect.

    Returns:
        list: A list of captured variables.
    """
    if not (callable(lambda_func) and  ((lambda_func.__name__ == "<lambda>") or (lambda_func.__name__ == "tf_multiply"))):
        raise ValueError("Provided input must be a lambda function.")

    # Access the closure
    closure = lambda_func.__closure__
    if not closure:
        return []  # Return None if there is no closure

    # Concatenate the cell contents
    concatenated = []
    for cell in closure:
        content = cell.cell_contents
        if isinstance(content, (list, tuple, set)):
            concatenated.extend(content)  # Extend with iterable contents
        else:
            concatenated.append(content)  # Append non-iterable contents
    return concatenated


def extract_function_and_args(lambda_func):
    """
    Extract the function and arguments from a lambda without executing it.

    Args:
        lambda_func (callable): The lambda function to inspect.

    Returns:
        tuple: A tuple containing the lambda's body (function) and its arguments.
    """
    if isinstance(lambda_func, tuple):
        return lambda_func  # Return the input if it is already a tuple

    if not (callable(lambda_func) and ((lambda_func.__name__ == "<lambda>") or (lambda_func.__name__ == "tf_multiply"))):
        raise ValueError("Provided input must be a lambda function.")

    # Retrieve closure variables
    closure_vars = get_closure_vars(lambda_func)

    # Inspect the function's signature
    signature = inspect.signature(lambda_func)
    arg_names = list(signature.parameters.keys())

    return lambda_func, closure_vars  # Return the function and captured variables


def apply_recursive_lambda(lambda_expr):
    """
    Recursively process a lambda function by extracting and applying its logic.

    Args:
        lambda_expr (lambda): The lambda function to process.

    Returns:
        The final result after applying the lambda function recursively.
    """
    func, args = extract_function_and_args(lambda_expr)

    # Process arguments recursively
    new_args = []
    for arg in args:
        if callable(arg):
            # Recursively process callable arguments
            res = apply_recursive_lambda(arg)
            new_args.append(res)
        else:
            # Non-callable arguments are added directly
            new_args.append(arg)
    # Recreate the lambda with updated arguments using functools.partial
    # This allows us to "bind" the modified arguments to the function
    #new_lambda = functools.partial(func, new_args)

    # Execute the modified lambda
    return func(new_args) #new_lambda()
