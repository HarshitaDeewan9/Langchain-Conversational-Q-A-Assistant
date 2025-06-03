from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Model you want to download
model_name = "sshleifer/tiny-gpt2"

# Path to current folder
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "tiny-gpt2")  # Folder will be ./tiny-gpt2

# Download and save model locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)

print(f"Model saved to: {model_dir}")
