
from jax import random
import jax

class KeyManager:
    """
    Manages the random number generation keys for JAX.
    Handles the generation and splitting of PRNG keys.
    """
    def __init__(self, seed):
        self.key = random.PRNGKey(seed)

    def get_key(self):
        """Ensure JAX doesn't track this key."""
        self.key, subkey = random.split(self.key)
        return jax.lax.stop_gradient(subkey) 

