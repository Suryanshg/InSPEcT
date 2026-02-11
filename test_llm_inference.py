import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from src.constants import DEVICE

# Init Model Path
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

# Init Soft Prompt Path 
soft_prompt_path = "trained_prompts/Meta-Llama-3-8B-Instruct_sst2_lr0.0008_8_epochs_pt_n56/epoch_0008_acc_0.957569.pt"

# Set a prompt
prompt = "Text: the passions aroused by the discord between old and new cultures are set against the strange , stark beauty of the mideast desert , so lovingly and perceptively filmed that you can almost taste the desiccated air . Label: First I should classify this as one of the following"
prev_generated_text = " Text: the passions aroused by the discord between old and new cultures are set against the strange, stark beauty of the mideast desert, so lovingly and perceptively filmed that you can almost taste the desiccated air. Label: First I should classify this as one of the following"

# Init Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Init Model
model = AutoModelForCausalLM.from_pretrained(model_path, dtype = torch.bfloat16).to(DEVICE)

# Init Peft Config
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.RANDOM, # Random Weight Init for Prompt Tuning Params
    num_virtual_tokens=56,
    tokenizer_name_or_path=model_path           # tokenizer to use
)

# Modify model to use Peft using the defined config
model = get_peft_model(model, peft_config)
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Init Soft Prompt Tokens
soft_prompt = torch.load(soft_prompt_path).to(model.dtype).to(DEVICE)

# Tokenize the prompt
tokenized_input = tokenizer(prompt, return_tensors="pt").to(DEVICE)
input_ids = tokenized_input['input_ids'].detach().clone()

# Determine the end token to stop
end_token = tokenizer.encode('$', add_special_tokens=False)[0]

# Do Forward Pass Thru the Model
with torch.no_grad():
    for _ in range(60):
        outputs = model(tokenized_input['input_ids'])
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token_id.squeeze(0)], dim=-1)
        if next_token_id.item() == end_token:
            break

generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens = True)
generated_text = "".join(generated_text)

print(generated_text)

assert generated_text == prev_generated_text