import tensorflow as tf
import functools
import inspect

from tensorflow.lookup.experimental import MutableHashTable

# Activate debug mode
DEBUG = 0

# Define constants
EPSILON=1e-7  #tf.keras.backend.epsilon()
TENSOR_EPSILON = 1e-6
ARE_PREC_SATISF_STR='ARE_PREC_SATISFIED'
LIFTED_STR="_LIFT_"

TF_ACTV_FN_BOOL = tf.nn.tanh
TF_ACTV_FN_REAL = tf.nn.relu
TF_ZERO=tf.constant(0.0, dtype=tf.float32)
TF_SAT=tf.constant(1.0, dtype=tf.float32)
MISSING_VALUE=-1 #tf.constant(-1.0, dtype=tf.float32)
TF_UN_SAT=tf.constant(-1.0, dtype=tf.float32)
NUM_ZERO=TF_ZERO.numpy() 
NUM_SAT=TF_SAT.numpy()
NUM_UN_SAT=TF_UN_SAT.numpy()

UNSAT_PENALTY=tf.constant(50000.0, dtype=tf.float32)
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

class GlobalData():
    """
    Class to store global data.
    """
  
    _class_lifted_actions_object_map={} # Map to store the up action and the corresponding lifted object 

    _class_lifted_actions_id_map={} # Map to store the up action and the corresponding ID

    _class_predicates_list_string=_class_predicates_list= [] #tf.Variable([], dtype=tf.string, size=0, dynamic_size=True)  # List to store the predicates
    tf_class_predicates_list= None
    _class_predicates_map=MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64, default_value=MISSING_VALUE) # Map to store the predicates
   
    _class_conditions_list=[]  # List to store the preconditions
    _class_conditions_map={} # Map to store the preconditions

    _class_effects_list=[]  # List to store the effects
    _class_effects_map={} # position of the effect in _effects_list



    def _insert_in_map(keys, table_kv, list_vk, value=None):
        assigned_values = []  # List to store the incremented values
        #table_kv=GlobalData._class_predicates_map
        #list_vk=GlobalData._class_predicates_list


        # Iterate over each key in the set
        for key_elem in keys:

            key = tf.convert_to_tensor(str(key_elem), dtype=tf.string)
             # Lookup current value
            current_value = table_kv.lookup(key)

            # Insert only if it's a new key
            if tf.equal(current_value, -1):
                if value is None:
                    current_value=table_kv.size()
                else:
                    current_value = value
                table_kv.insert(key, current_value )
                current_value = table_kv.lookup(key)
                list_vk.insert(current_value, str(key_elem))

            assigned_values.append(current_value.numpy())  # Store result

        # Return the tensor of incremented values
        return tf.constant(assigned_values, dtype=tf.int32)
        #return tf.stack(assigned_values)

    def _get_keys_from_list(indexes, list_vk):
        '''
        Retrieves keys corresponding to the given indexes from the corresponding indexes
        '''
        keys=[]  
        for indx in indexes:
            if indx<0 or indx>=len(list_vk):
                key=None
            else:
                key = list_vk[indx]
            keys.append(key)
        return keys



    def _get_values_from_table(indexes, table_kv):
        '''
        Retrieves keys corresponding to the given indexes from the corresponding indexes
        '''
        keys=[]  
        for indx in indexes:
            keys.append(table_kv.lookup(str(indx)))
        return keys
    
    def insert_predicates_in_map(keys):
        
        table_kv=GlobalData._class_predicates_map
        list_vk=GlobalData._class_predicates_list

        return GlobalData._insert_in_map(keys, table_kv, list_vk)

    
    def get_key_from_indx_predicates_list(indx):
        '''
        Retrieves keys corresponding to the given indexes from the corresponding indexes
        '''
        list_vk=GlobalData.tf_class_predicates_list
        key = list_vk[indx]
        
        return key

    def get_keys_from_predicates_list(indexes):
        '''
        Retrieves keys corresponding to the given indexes from the corresponding indexes
        '''
        list_vk=GlobalData.tf_class_predicates_list
        return GlobalData._get_keys_from_list(indexes, list_vk)

    def get_values_from_predicates_list(indexes):
        '''
        Retrieves keys corresponding to the given indexes from the corresponding indexes
        '''
        table_kv=GlobalData._class_predicates_map
        return GlobalData._get_values_from_table(indexes, table_kv)

    def set(self, key, value):
        """
        Set a global value.

        Args:
            key (str): The key to set.
            value (object): The value to set.
        """
        self._data[key] = value

    def get(self, key):
        """
        Get a global value.

        Args:
            key (str): The key to get.

        Returns:
            object: The value associated with the key.
        """
        return self._data[key]


     
    def get_lifted_string(prec_str:str, predicates_set, lifted_str:str=LIFTED_STR):
        ''' Replace the predicates in the string with the lifted string+index'''
        for indx, pred in enumerate(predicates_set):
            prec_str=prec_str.replace(str(pred), lifted_str+str(indx))
        return prec_str
   

    def __repr__(self):
        return str(self._data)

    def __str__(self):
        return str(self._data)