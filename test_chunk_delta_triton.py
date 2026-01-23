
import sys
import os
os.environ["TRITON_CPU_BACKEND"] = "1"
os.environ["TRITON_INTERPRET"] = "1"
import torch

# Add the 'fla' directory to sys.path
# The current directory is /home/askr/Documents/github/tpu-research
# The fla package is in /home/askr/Documents/github/tpu-research/fla
fla_path = os.path.abspath(os.path.join(os.getcwd(), 'fla'))
sys.path.append(fla_path)

try:
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    print("Successfully imported chunk_gated_delta_rule_fwd_h")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_triton_run():
    device = torch.device('cpu')
    torch.manual_seed(42)
    
    # Dimensions
    B, T, H, K, V = 2, 128, 4, 32, 64
    chunk_size = 64
    
    # Initialize inputs on GPU
    # k: [B, T, H, K]
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    # w: [B, T, H, K]
    w = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    # u: [B, T, H, V]
    u = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
    # g: [B, T, H] -> Need to be consistent with kernel expectation (cumulative or not?)
    # In chunk_delta_h.py, it expects g.
    # Usually passed as [B, T, H].
    g = torch.randn(B, T, H, device=device, dtype=torch.float32)
    
    h0 = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
    
    print("Running Triton implementation on GPU...")
    # Call the function
    # Note: args are k, w, u, g, gk, initial_state, output_final_state...
    h_out, v_new_out, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
        use_exp2=True # The code uses exp2 by default logic often
    )
    
    print("Run complete.")
    print(f"Output shapes:")
    print(f"  h (History): {h_out.shape}")
    if v_new_out is not None:
        print(f"  v_new (Residual): {v_new_out.shape}")
    print(f"  final_state: {final_state.shape}")
    
    print("\nSample stats:")
    print(f"  Final State mean: {final_state.mean().item():.6f}")

if __name__ == "__main__":
    test_triton_run()
