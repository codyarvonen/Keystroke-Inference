import torch
from transformers import GPT2LMHeadModel, GPT2Config

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Configuration & Model Loading
print("Loading GPT-2 Small...")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config).to(device)

# 3. Project Logic: Freeze GPT-2 Weights
for param in model.parameters():
    param.requires_grad = False

# 4. Simulation Parameters
prefix_length = 50 
text_length = 50 
batch_size = 8  
total_length = prefix_length + text_length

# 5. Trainable Component Simulation
trainable_prefix = torch.randn(batch_size, prefix_length, config.n_embd, 
                               requires_grad=True, device=device)

# 6. Parameter Counting Logic
def count_parameters(model, trainable_tensor):
    frozen_params = sum(p.numel() for p in model.parameters())
    trainable_params = trainable_tensor.numel() 
    return frozen_params, trainable_params

frozen, trainable = count_parameters(model, trainable_prefix)

print("-" * 30)
print(f"MODEL STATISTICS:")
print(f"Frozen Parameters:    {frozen:,}")
print(f"Trainable Parameters: {trainable:,} (Prefix Bridge)")
print(f"Total Parameters:     {frozen + trainable:,}")
print("-" * 30)

# 7. Labels with Masking (-100)
# We mask the prefix so the loss is only calculated on the generated text.
prefix_labels = torch.full((batch_size, prefix_length), -100, dtype=torch.long, device=device)
text_labels = torch.randint(0, config.vocab_size, (batch_size, text_length), device=device)
full_labels = torch.cat((prefix_labels, text_labels), dim=1)

# 8. Memory Stress Test
print(f"Running backpropagation test on {torch.cuda.get_device_name(0)}...")

try:
    torch.cuda.reset_peak_memory_stats()
    
    # Forward Pass
    # We retrieve embeddings for text and concatenate with our trainable IMU prefix.
    inputs_embeds = model.transformer.wte(text_labels) 
    full_embeddings = torch.cat((trainable_prefix, inputs_embeds), dim=1)
    
    # Passing full_labels (length 100) to match the concatenated input sequence.
    outputs = model(inputs_embeds=full_embeddings, labels=full_labels)
    loss = outputs.loss
    
    # 9. Backward Pass
    # This evaluates if 12GB can hold the activations for backprop.
    loss.backward()
    
    # Usage Statistics
    max_memory = torch.cuda.max_memory_allocated() / 1024**2
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
    
    print("✅ TEST PASSED: Labels Aligned and Memory Sufficient.")
    print(f"Peak VRAM Usage: {max_memory:.2f} MiB")
    print(f"Available VRAM:  {total_memory:.2f} MiB ({(max_memory / total_memory) * 100:.1f}% utilized)")

except RuntimeError as e:
    if "out of memory" in str(e):
        print("❌ OOM Error: Current settings exceed 12GB VRAM.")
    else:
        print(f"Error: {e}")