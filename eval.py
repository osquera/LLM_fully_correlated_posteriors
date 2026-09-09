import torch
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup Base Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
model.eval()

def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[offset : offset + numel].view_as(p))
            offset += numel

# 2. Define the Evaluation Prompt
prompt = "Question: What is 2 + 2?\nAnswer: "
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_length = inputs.input_ids.shape[1]
max_tokens = 5  # Generate just enough tokens to see the core answer

# 3. Load Samples and Evaluate
sample_files = sorted(glob.glob("posterior_samples/sample_*.pt"))

if not sample_files:
    print("No samples found! Make sure you run the sampling script first.")

for file_index, file_path in enumerate(sample_files):
    print(f"\n{'='*60}")
    print(f"Model Sample {file_index + 1}: {file_path}")
    print(f"{'='*60}")
    
    # Load and inject weights
    theta_sample = torch.load(file_path, map_location=device)
    set_flat_params(model, theta_sample)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Isolate only the newly generated tokens
    gen_sequences = outputs.sequences[0, input_length:]
    
    # Loop through each generated step
    for step, token_id in enumerate(gen_sequences):
        # Convert raw logits to probabilities
        logits = outputs.scores[step][0]
        probs = torch.softmax(logits, dim=-1)
        
        # Identity and probability of the selected token
        selected_token_str = repr(tokenizer.decode(token_id))
        selected_prob = probs[token_id].item()
        
        # Top 5 candidates for this step
        top5_probs, top5_indices = torch.topk(probs, 5)
        
        print(f"\nStep {step + 1} | Selected Token: {selected_token_str} (Prob: **{selected_prob:.2%}**)")
        print("-" * 50)
        
        for rank in range(5):
            cand_id = top5_indices[rank]
            cand_prob = top5_probs[rank].item()
            cand_str = repr(tokenizer.decode(cand_id))
            print(f"  {rank + 1}. {cand_str:<15} : {cand_prob:>7.2%}")