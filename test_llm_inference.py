import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.constants import DEVICE

# Init Model Path
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

# Init Soft Prompt Path 
soft_prompt_path = "to_patch/epoch_0006_acc_0.947000_01.pt"

# Set a prompt
prompt = "| First I should classify this as:"

# Init Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Init Model
model = AutoModelForCausalLM.from_pretrained(model_path, dtype = torch.bfloat16).to(DEVICE)

# Set the padding token for the model
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Get the word embeddings of the model
word_embeddings = model.get_input_embeddings()

# Init Soft Prompt Tokens
soft_prompt = torch.load(soft_prompt_path).to(model.dtype).to(DEVICE).unsqueeze(0)

# Tokenize the prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

# Embed the Tokenized Prompt
input_embed = word_embeddings(input_ids).to(dtype = model.dtype)

# Add the Soft Prompt to the Input Prompt Embeddings
input_embed = torch.cat([soft_prompt, input_embed], dim = 1)

# Prepare an arbitrary end token (to trick the model.generate() function to generate until max_new_tokens)
end_token = tokenizer.encode('$', add_special_tokens=False)[0]

# Prepare attention mask
attention_mask = torch.ones(input_embed.size()[:-1], 
                            device=input_embed.device, 
                            dtype=torch.long)

# Perform Generation using the model
output_ids = model.generate(
    inputs_embeds=input_embed,
    attention_mask=attention_mask,
    max_new_tokens=60,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=end_token
)

# # Do Forward Pass Thru the Model
# with torch.no_grad():
#     for _ in range(60):
#         outputs = model(tokenized_input['input_ids'])
#         logits = outputs.logits[:, -1, :]
#         next_token_id = torch.argmax(logits, dim=-1)
#         input_ids = torch.cat([input_ids, next_token_id.squeeze(0)], dim=-1)
#         if next_token_id.item() == end_token:
#             break

# Decode the generated text
generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
generated_text = "".join(generated_text)

print(f"GENERATED TEXT: {generated_text}")