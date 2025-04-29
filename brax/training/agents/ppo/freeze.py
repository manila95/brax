import flax
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from typing import Any, Dict

# Create mask function that identifies the last layer without relying on names
def create_trainable_mask(params: Dict) -> Dict:
    """
    Create a mask dictionary that has the same structure as the parameters dict.
    Only parameters in the last layer will be True (trainable).
    """
    flat_params = flax.traverse_util.flatten_dict(params)
    
    # Group parameters by their module path (everything except the parameter type)
    path_to_keys = {}
    for k in flat_params.keys():
        # The path is everything except the last element (which is typically 'kernel' or 'bias')
        path = k[:-1]
        if path not in path_to_keys:
            path_to_keys[path] = []
        path_to_keys[path].append(k)
    
    # Sort paths by depth - the last layer typically has the maximum depth
    # If multiple layers at the same depth, we can use lexicographical ordering
    sorted_paths = sorted(path_to_keys.keys(), key=lambda x: (len(x), x))
    
    # The last layer is the last in the sorted list
    last_layer_path = sorted_paths[-1]
    
    # Create mask: True for the last layer parameters, False for everything else
    trainable_mask = {k: k[:-1] == last_layer_path for k in flat_params.keys()}
    
    return flax.traverse_util.unflatten_dict(trainable_mask)

# Alternative approach: mask by parameter depth
def create_trainable_mask_by_depth(params: Dict) -> Dict:
    """
    Create a mask where only the deepest parameters (likely the last layer) are trainable.
    """
    flat_params = flax.traverse_util.flatten_dict(params)
    
    # Find the maximum depth of any parameter
    max_depth = max(len(k) for k in flat_params.keys())
    
    # Create mask where only parameters at the maximum depth are trainable
    trainable_mask = {k: len(k) == max_depth for k in flat_params.keys()}
    
    return flax.traverse_util.unflatten_dict(trainable_mask)

# Function to identify the output layer in common architectures
def create_trainable_mask_for_output_layer(params: Dict) -> Dict:
    """
    Attempts to identify the output layer based on common naming patterns
    and layer structures in neural networks.
    """
    flat_params = flax.traverse_util.flatten_dict(params)
    
    # Check for common output layer names
    output_layer_names = ['output', 'logits', 'head', 'classifier', 'fc', 'predictions']
    
    trainable_mask = {}
    for k in flat_params.keys():
        is_output_layer = False
        
        # Check if any part of the key contains output layer names
        path_str = ''.join(str(p) for p in k)
        if any(name in path_str.lower() for name in output_layer_names):
            is_output_layer = True
        
        trainable_mask[k] = is_output_layer
    
    # If no output layer was found by name, fall back to the depth-based approach
    if not any(trainable_mask.values()):
        return create_trainable_mask_by_depth(params)
    
    return flax.traverse_util.unflatten_dict(trainable_mask)