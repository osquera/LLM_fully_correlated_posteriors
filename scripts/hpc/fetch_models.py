"""Download models into the HF cache on a login node (compute nodes may not
reach the hub): uv run python scripts/hpc/fetch_models.py Qwen/Qwen2.5-0.5B ..."""
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

for name in sys.argv[1:]:
    AutoTokenizer.from_pretrained(name)
    AutoModelForCausalLM.from_pretrained(name)
    print("cached", name)
