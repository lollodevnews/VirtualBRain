import torch

class VBRQuantumEmulator:
    def __init__(self, num_qubits):
        """
        Initializes the quantum register.
        State 01 (Integer 1) represents Superposition (Processing).
        """
        self.num_qubits = num_qubits
        # Initialize all qubits in state '01' (Processing / Superposition)
        self.register = torch.ones(num_qubits, dtype=torch.uint8) 
        
    def show_state(self):
        """Translates the 2-bit integer state into human-readable quantum states."""
        state_map = {0: "[00] Invalid/Dust", 1: "[01] Superposition", 2: "[10] Qubit |0>", 3: "[11] Qubit |1>"}
        return [state_map[int(q)] for q in self.register]

    def apply_oracle(self, target_factor):
        """
        The Quantum Oracle. 
        In a real quantum computer, this applies a phase flip to the correct answer.
        In our VBR emulator, we use standard tensor math to 'mark' the correct binary state.
        """
        print(f"\n--- Applying Factorization Oracle for target: {target_factor} ---")
        possible_states = torch.arange(2**self.num_qubits)
        
        # 1. Create a safe mask to avoid universe-breaking ZeroDivisionErrors
        valid_mask = (possible_states > 1) & (possible_states < target_factor)
        
        # 2. Initialize an empty boolean tensor for our states
        is_correct_factor = torch.zeros_like(possible_states, dtype=torch.bool)
        
        # 3. Only perform the modulo division on the safe, valid states
        is_correct_factor[valid_mask] = (target_factor % possible_states[valid_mask] == 0)
        
        # We save the correct binary answer as our "Interference Pattern"
        valid_indices = torch.nonzero(is_correct_factor, as_tuple=True)[0]
        
        if valid_indices.numel() > 0:
            self.interference_pattern = valid_indices[0] # Pick the first valid factor
        else:
            self.interference_pattern = None

    def measure(self):
        """
        The Wave Collapse (Measurement).
        Forces the '01' superposition states to jump to '10', '11', or '00'.
        """
        print("\n--- Collapsing Wave Function (Measurement) ---")
        
        if self.interference_pattern is None:
            # Complete Decoherence: The Oracle found no answer.
            # State jumps from '01' (Superposition) to '00' (Invalid/Dust)
            self.register = torch.zeros(self.num_qubits, dtype=torch.uint8)
            return
            
        # Convert the correct answer into a binary tensor (e.g., 3 -> [0, 1, 1])
        binary_string = format(self.interference_pattern.item(), f'0{self.num_qubits}b')
        classical_bits = torch.tensor([int(b) for b in binary_string], dtype=torch.uint8)
        
        # Map classical binary (0, 1) to our VBR collapsed states ('10', '11')
        # If bit is 0 -> State 2 ('10')
        # If bit is 1 -> State 3 ('11')
        collapsed_state = classical_bits + 2
        
        self.register = collapsed_state

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # We want to factor the number 15. We need 3 qubits to represent numbers up to 7.
    qpu = VBRQuantumEmulator(num_qubits=3)
    
    print("1. Initialization (Register in Superposition):")
    print(qpu.show_state())
    
    # Send the problem to the Oracle while the register is in state '01'
    qpu.apply_oracle(target_factor=15)
    
    print("\n2. Status Check (During Oracle Operation):")
    # It still outputs '01' because we haven't measured it yet!
    print(qpu.show_state()) 
    
    # Force the collapse
    qpu.measure()
    
    print("3. Final Measurement (Wave Collapsed):")
    print(qpu.show_state())
