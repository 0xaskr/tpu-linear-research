
import torch
import sys
import os

# Add the fla directory to python path if not installed
sys.path.append(os.path.join(os.getcwd(), 'fla'))

from fla.layers.kda import KimiDeltaAttention

def test_kda_run():
    print("Initializing KimiDeltaAttention...")
    
    # Setup parameters
    batch_size = 2
    seq_len = 128
    hidden_size = 1024
    num_heads = 8
    head_dim = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Layer
    # mode='chunk' is for training (parallel)
    model = KimiDeltaAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        mode='chunk'
    ).to(device)
    

    # Create dummy input
    x = torch.randn(batch_size, seq_len, hidden_size).to(device)
    
    # 构造 Attention Mask (Batch, Seq) 用于演示 Padding 处理
    # 样本 0: 全长有效 (全是 1)
    # 样本 1: 后半段是 padding (置为 0)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    attention_mask[1, seq_len // 2:] = 0
    
    print(f"Input shape: {x.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Forward pass
    print("Running forward pass (chunk mode)...")
    output, _, _ = model(x, attention_mask=attention_mask)
    
    print(f"Output shape: {output.shape}")
    
    # Inference mode (recurrent)
    print("\nRunning inference mode (recurrent)...")
    model.mode = 'fused_recurrent'
    # Short sequences often trigger recurrent mode automatically in forward(), 
    # but we force it here by setting the property if needed, 
    # though the class uses self.mode.
    
    # Note: KimiDeltaAttention's forward method has logic to switch to 'fused_recurrent' 
    # automatically for short sequences during inference (not training).
    model.eval() 
    with torch.no_grad():
        # inference 模式同样支持 mask (尽管通常 inference 是逐 token 生成，这里演示的是 prefill 或 batch inference)
        output_inf, _, _ = model(x, attention_mask=attention_mask)
        print(f"Output shape (inference): {output_inf.shape}")
        
    print("\nSuccess! The KDA kernel is running.")

if __name__ == "__main__":
    test_kda_run()
