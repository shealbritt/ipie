
from jax import random

class KeyManager:
    """
    Manages the random number generation keys for JAX.
    Handles the generation and splitting of PRNG keys.
    """
    def __init__(self, seed):
        self.key = random.PRNGKey(seed)
        self.index = 0
    
    def _get_next_key(self):
        """Splits the key and returns a new subkey."""
        new_key, self.key = random.split(self.key)
        return new_key

    def get_key(self):
        """Public method to retrieve a new PRNG key."""
        return self._get_next_key()
    

if "__main__":
    keymanager = KeyManager(1)
    print(keymanager.get_key())