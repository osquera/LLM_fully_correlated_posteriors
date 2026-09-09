import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

# 1. Load SmolLM2-135M
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
model.eval()

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])

def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[offset : offset + numel].view_as(p))
            offset += numel

theta_map = get_flat_params(model)
P = theta_map.numel()
print(f"Total trainable parameters: {P:,}")


def compute_sample_loss_grad(model, input_ids, labels):
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
    return grad.detach()


# Format tiny_qa_benchmark into (input_ids, labels)
dataset = load_dataset("vincentkoc/tiny_qa_benchmark", split="train")
processed_samples = []

for item in dataset:
    prompt = f"Question: {item['text']}\nAnswer: "
    answer = f"{item['label']}\n"
    
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    
    input_ids = torch.tensor([prompt_ids + answer_ids], device=device)
    labels = torch.tensor([[-100] * len(prompt_ids) + answer_ids], device=device)
    processed_samples.append((input_ids, labels))

S = 8  # Projection batch size
batches = [processed_samples[i : i + S] for i in range(0, len(processed_samples), S)]

precomputed_batches = []
print(f"Precomputing gradients across {len(batches)} batches...")

for b_idx, batch in enumerate(batches):
    grad_list = []
    for input_ids, labels in batch:
        g = compute_sample_loss_grad(model, input_ids, labels)
        grad_list.append(g)
    
    # M_b: (S x P) - Upcast to float32 for stable math
    M_b = torch.stack(grad_list).to(torch.float32)
    
    # Gram matrix: (S x S)
    Gram = M_b @ M_b.T
    
    # Add a tiny damping factor for numerical stability
    damping = 1e-6 * torch.eye(Gram.size(0), device=device, dtype=torch.float32)
    Gram_damped = Gram + damping
    
    precomputed_batches.append({
        "M_b": M_b,               
        "Gram_damped": Gram_damped 
    })

def sample_loss_projected_posterior(theta_map, precomputed_batches, alpha=100000, t_max=15):
    """
    Simulates: theta ~ N(theta_map, alpha^(-1) * U_L U_L^T)
    """
    # 1. Sample isotropic noise: epsilon ~ N(0, alpha^(-1) * I)
    std = (1.0 / alpha) ** 0.5
    delta = torch.randn(theta_map.shape, device=theta_map.device, dtype=torch.float32) * std

    # 2. Alternating projections loop
    for iteration in range(t_max):
        for batch in precomputed_batches:
            M_b = batch["M_b"]
            Gram_damped = batch["Gram_damped"]
            
            # v = M_b @ delta (delta should also be float32 during this loop)
            v = torch.mv(M_b, delta)
            
            # Solve (M_b M_b^T) w = v instead of inverting
            w = torch.linalg.solve(Gram_damped, v)
            
            # delta = delta - M_b^T @ w
            delta = delta - torch.mv(M_b.T, w)
            
    return theta_map.to(torch.float32) + delta

# Sample 5 posterior models
num_samples = 5
posterior_samples = []

for s in range(num_samples):
    theta_sample = sample_loss_projected_posterior(theta_map, precomputed_batches, alpha=100000.0, t_max=20)
    posterior_samples.append(theta_sample)


import os

output_dir = "posterior_samples"
os.makedirs(output_dir, exist_ok=True)

print(f"Saving {num_samples} posterior samples to '{output_dir}/'...")
for i, theta_sample in enumerate(posterior_samples):
    # Save the flat tensor to disk
    torch.save(theta_sample.cpu(), f"{output_dir}/sample_{i}.pt")

print("Done! You can now run your evaluation script separately.")

# Evaluate a sample
set_flat_params(model, posterior_samples[0])

test_prompt = "Question: What is the capital of France?\nAnswer: "
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))