# Lesson 7: Inference & Real-World Use Cases

## 1. Introduction

### Learning Objectives

By the end of this lesson, you will be able to:

- Load the fine-tuned model and adapters for inference
- Run the model in "Interactive Mode" to generate commands
- Understand key use cases for SecureCLI-Tuner in enterprise environments
- Integrate the model into DevOps workflows

> [!NOTE]
> **VERIFIED DATA:** The examples in this lesson work with the `honest-music-2` checkpoint.

### Why Inference Matters

Training is only half the battle. To be useful, the model must be deployable and accessible to users. In this lesson, we shift from "builder" mode to "user" mode.

---

## 2. Running Inference

To use the model, we need to load:

1. **Base Model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (in 4-bit precision)
2. **LoRA Adapters:** Our fine-tuned weights (from `honest-music-2`)

### Inference Script (`scripts/inference.py`)

Create a script to load the model and generate responses:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Load Base Model
base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)

# 2. Load Fine-Tuned Adapters
adapter_path = "models/checkpoints/final_checkpoint" # or your huggingface repo
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. Generate
def generate_command(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful DevOps assistant. Generate a Bash command for the given instruction."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Extract just the assistant response
    return response.split("assistant\n")[-1].strip()

# Test
print(generate_command("Show me the size of all log files in /var/log"))
```

---

## 3. Interactive Mode

For testing, an interactive loop is useful.

```python
# scripts/interactive.py

print("SecureCLI-Tuner V2 Interactive Mode")
print("Type 'exit' to quit.\n")

while True:
    user_input = input(">> ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    try:
        command = generate_command(user_input)
        
        # Run CommandRisk validation here (simulated)
        if "rm -rf /" in command:
            print("❌ BLOCKED: Dangerous command detected")
        else:
            print(f"✅ Safe Command: {command}")
            
    except Exception as e:
        print(f"Error: {e}")
```

---

## 4. Real-World Use Cases

How does this model fit into a real engineering organization?

### Use Case 1: The "Local DevOps Assistant"

**Who:** Junior Developers, Data Scientists
**Problem:** Remembering complex `kubectl`, `docker`, or `git` syntax.
**Solution:** A CLI tool (`secure-cli "find large files"`) that gives the answer instantly without context switching to a browser.
**Value:** Speed + Safety (prevents copy-pasting dangerous commands from StackOverflow).

### Use Case 2: CI/CD Pipeline Validator

**Who:** DevOps Engineers
**Problem:** Reviewing hundreds of lines of shell scripts in Pull Requests.
**Solution:** The **CommandRisk Engine** (Lesson 4) can be run strictly as a validator. It scans shell scripts in PRs for blocked patterns (Layer 1) or anomalous logic (Layer 2).
**Value:** Automated security governance for infrastructure-as-code.

### Use Case 3: Educational "Sandbox"

**Who:** Students, New Hires
**Problem:** Learning Linux usage without breaking the system.
**Solution:** A web interface where students type intents ("wipe the disk") and the model explains *why* the generated command (`dd if=/dev/zero...`) is dangerous and blocks it.
**Value:** Safe learning environment with immediate feedback.

---

## 5. Deployment Options

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Local (Quantized)** | Free, Private | Requires GPU/RAM | Individual Developers |
| **RunPod / Cloud** | Scalable, Fast | Cost ($/hr) | Teams / Org-wide Service |
| **HuggingFace API** | Easy Setup | Rate Limits, Public | Demos / Prototyping |

---

## 6. Interview Preparation

### Q: How would you deploy this for a team of 50 engineers?

**Model Answer:** "I would deploy the model as an internal API service using vLLM or TGI on a GPU instance (like an A10G). Developers would install a lightweight CLI client that sends queries to this API. This keeps the heavy compute centralized and ensures everyone benefits from the same security guardrails. I'd enable the CommandRisk engine on the API server to guarantee that no unsafe commands are returned, regardless of the prompt."

### Q: Can this model replace StackOverflow?

**Model Answer:** "For specific command syntax, yes. It's faster and more secure because of our fine-tuning and guardrails. However, for architectural decisions or debugging complex errors, developers still need external resources. This tool focuses on the 'how-to' of CLI operations, not the 'why' of system design."

---

## 7. Key Takeaways

- ✅ Inference requires loading base model + LoRA adapters
- ✅ Interactive mode enables rapid feedback loops
- ✅ Use cases range from personal assistants to automated security gates
- ✅ Centralized deployment (API) is best for teams

---

## 8. Final Words

Congratulations! You've completed the **SecureCLI-Tuner** course. You've built a security-first LLM system from scratch:

1. **Data:** Cleaned and filtered for safety
2. **Training:** Fine-tuned with QLoRA
3. **Evaluation:** Validated with semantic metrics & adversarial tests
4. **Guardrails:** Protected by the CommandRisk engine

You are now ready to build safe, agentic AI tools.
