import torch
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load base model (MAP)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
model.eval()

# Helper to swap weights
def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[offset : offset + numel].view_as(p))
            offset += numel
    return model

# 1. Generate the baseline answer using the MAP model
prompt = "Question: What is 2 + 2?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
prompt_len = inputs.input_ids.shape[1]

with torch.no_grad():
    map_outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    
generated_text = tokenizer.decode(map_outputs[0, prompt_len:], skip_special_tokens=True)
print(f"Base Generation: '{generated_text}'\n")

# 2. Gather logits from all 5 posterior samples for this exact sequence
sample_files = sorted(glob.glob("posterior_samples/sample_*.pt"))
all_sample_logits = []

for file_path in sample_files:
    # Load perturbed weights
    theta_sample = torch.load(file_path, map_location=device)
    set_flat_params(model, theta_sample)
    
    with torch.no_grad():
        # Forward pass with the full sequence (prompt + generated tokens)
        outputs = model(map_outputs)
        # Extract logits corresponding to the generated tokens
        # Shift by 1: logits at index i predict token at index i+1
        gen_logits = outputs.logits[0, prompt_len-1 : -1, :] 
        all_sample_logits.append(gen_logits)

# 3. Calculate Uncertainty (Maximum Logit Variance)
# Stack logits: shape -> (num_samples, num_generated_tokens, vocab_size)
stacked_logits = torch.stack(all_sample_logits).to(torch.float32)

# Variance across the 5 models: shape -> (num_generated_tokens, vocab_size)
logit_variance = torch.var(stacked_logits, dim=0)

# Max variance across vocabulary dimensions (as per the paper)
max_variance_per_token = torch.max(logit_variance, dim=-1).values

# Aggregate for the whole sequence
mean_uncertainty = torch.mean(max_variance_per_token).item()

print("-" * 50)
print(f"Sequence Uncertainty Score (Max Logit Variance): {mean_uncertainty:.4f}")
for i, token_id in enumerate(map_outputs[0, prompt_len:]):
    token_str = repr(tokenizer.decode(token_id))
    print(f"Token: {token_str:<10} | Max Variance: {max_variance_per_token[i].item():.4f}")