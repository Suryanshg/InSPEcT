import torch
from src.constants import DEVICE
import matplotlib.pyplot as plt
import numpy as np


def set_hs_patch_hooks(model, hs_patch_config, num_of_tokens):
    def patch_hs(name, position_hs):
        def hook(module, input, output):
            for position_, hs_ in position_hs:
                # (batch, sequence, hidden_state)
                # output[0][0, position_ : position_ + num_of_tokens] = hs_ #NOTE: A Bug Maybe, as it is straightforwardly incompatible...
                output[0][position_ : position_ + num_of_tokens] = hs_
        return hook

    hooks = []
    for l in hs_patch_config:
        if model.config.model_type == 'llama':
            if l == -1:
                layer = model.model.embed_tokens
            else:
                layer = model.model.layers[l]

        else:
            raise ValueError(f"Unknown model: {model.config.model_type}")

        hooks.append(layer.register_forward_hook(
            patch_hs(f"patch_hs_{l}", hs_patch_config[l])
        ))

    return hooks


def set_soft_prompt_patch_hook(model, soft_prompt, source_position, num_of_tokens):
    # Create a hook that replaces hidden states at source_position with soft_prompt tensor
    def patch_sp(name, soft_prompt):
        def hook(module, input, output):
            # (batch, sequence, hidden_state)
            if model.config.model_type == "llama":
                # output[0][0, source_position : source_position + num_of_tokens] = soft_prompt     #NOTE: A Bug Maybe, as it is straightforwardly incompatible...
                output[0][source_position : source_position + num_of_tokens] = soft_prompt
            
            else:
                raise ValueError(f"Unknown model: {model.config.model_type}")
        
        return hook
    
    # If its a Llama model
    if model.config.model_type == "llama":

        # Fetch the first layer
        first_layer: torch.nn.Module = model.model.layers[0]

        # Fetch the embedding layer
        embed_layer: torch.nn.Module = model.model.embed_tokens
    
    # Raise Error if model is any other than a Llama
    else:
        raise ValueError(f"Unknown model: {model.config.model_type}")
    
    # Attach the hook to layer 1 (first decoder layer) (so soft_prompt enters at the start of processing)
    # hooks = [first_layer.register_forward_hook(
    #     patch_sp("patch_sp_0", soft_prompt)
    # )]

    # Attach the hook to layer 0 (embed_tokens layer) (so soft_prompt enters at the start of processing)
    hooks = [embed_layer.register_forward_hook(
        patch_sp("patch_sp_0", soft_prompt)
    )]

    # Return all the created hooks (only 1 here though)
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def build_soft_hs_cache(soft_prompt, model, tokenizer, num_of_tokens):
    # Create a list [0, 1, 2, ..., num_layers] - All layers including embeddings
    layers_to_cache = list(range(model.config.num_hidden_layers+1))

    # Init Hidden State Cache as a dict
    hs_cache = {}

    # Tokenize Placeholder text like " x x x" (example for 3 tokens)
    inp = tokenizer(" x" * num_of_tokens, return_tensors="pt").to(DEVICE)

    # Calculate starting position of the "x" tokens in the sequence.
    pos = inp['input_ids'].shape[1] - num_of_tokens

    # Convert soft_prompt to model's dtype (bfloat16)
    soft_prompt = soft_prompt.to(model.dtype)
    
    # Inject soft_prompt at layer 0
    patch_hooks = set_soft_prompt_patch_hook(model, soft_prompt, pos, num_of_tokens) 

    # Run forward pass of the placeholder tokens
    with torch.no_grad():
        output = model(**inp, output_hidden_states = True)

    # Remove all hooks (call-back methods after each forward call on a layer)
    remove_hooks(patch_hooks)

    # Debug: Check if soft_prompt was injected at embedding layer
    # print(f"Checking layer 0 (embeddings):")
    # print(f"Match: {torch.allclose(output['hidden_states'][0][0][pos:], soft_prompt, atol=1e-5)}")
    # print(f"Max diff: {(output['hidden_states'][0][0][pos:] - soft_prompt).abs().max().item()}")

    # For each layer idx in layers to cache
    for layer in layers_to_cache:

        # If layer is not in cache yet
        if layer not in hs_cache:

            # Allocate an empty list for that layer
            hs_cache[layer] = []

        # Store hidden state at the patched position
        hs_cache[layer].append(output["hidden_states"][layer][0][pos:])

    # Save the hs_cache
    # torch.save(hs_cache, 'hs_cache_first_layer.pt')

    # Return the hidden state cache and the tokenized placeholder text
    return hs_cache, inp


def generate_greedy_deterministic(hs_patch_config, 
                                  inp, 
                                  max_length, 
                                  end_token, 
                                  model, 
                                  tokenizer, 
                                  num_of_tokens,
                                  source_layer, 
                                  target_layer,
                                  do_sample = False,
                                  temperature = 1.0,
                                  visualize_confidence = False):
    
    # Copy target prompt's input token ids for generation (IDK why again)
    input_ids = inp["input_ids"].detach().clone().to(DEVICE)

    # Without this, we mostly get warnings
    # model.set_attn_implementation('eager')

    # Track data for visualization
    viz_data = []

    with torch.no_grad():
        for step in range(max_length):
            patch_hooks = set_hs_patch_hooks(model, hs_patch_config, num_of_tokens) 
            outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
            remove_hooks(patch_hooks)
            
            # Extract logits for the last token
            # output.logits has shape                       # (1, seq_len, vocab_size)
            logits = outputs.logits[:, -1, :]               # (1, vocab_size)

            # Compute probablilities (before temp scaling)
            # Apply softmax to the last dim (which is vocab dim)
            raw_probs = torch.softmax(logits, dim = -1)     # (1, vocab_size)

            # Get the next token id using the logits
            if do_sample:
                # Apply temperature scaling
                logits = logits / temperature

                # Sample from the softmax distribution
                probs = torch.softmax(logits, dim = -1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1) # (1,)

            else:
                next_token_id = torch.argmax(logits, dim=-1)                       # (1, )

            # Collect data for visualizing confidence
            if visualize_confidence:
                # Decode the next token id
                chosen_token = tokenizer.decode([next_token_id.item()])
                
                # Get probability of the chosen token
                # Choose the first batch (0 idx) and the token's prob using the token's id as the idx
                chosen_prob = raw_probs[0, next_token_id.item()].item()

                viz_data.append({
                    "step": step,
                    "chosen_token": chosen_token,
                    "chosen_prob": chosen_prob,

                    # Compute Entropy
                    # H = -Σ p(token) * log(p(token))
                    "entropy": -(raw_probs * torch.log(raw_probs + 1e-10)).sum(dim = -1).item()
                })


            # Concat the next token id to the input_ids for autoregressive generation
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            # If the end_token is predicted, break out of the autoregression loop
            if next_token_id.item() == end_token:
                break

    # Generate Confidence Visualization if needed
    if visualize_confidence and viz_data:
        generate_confidence_visualizations(viz_data, source_layer, target_layer)

    generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    patched_pattern = (" x" * num_of_tokens).strip()
    return "".join(generated_text).split(patched_pattern)[-1] 


def generate_confidence_visualizations(viz_data, source_layer, target_layer):

    # Create two subplots in two rows
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Extract individual components from viz_data
    steps = [s["step"] for s in viz_data]
    chosen_probs = [s["chosen_prob"] for s in viz_data]
    chosen_tokens = [repr(s["chosen_token"]) for s in viz_data]
    entropies = [s["entropy"] for s in viz_data]

    # Plot Chosen next token probability per step
    # bars = axes[0].bar(steps, chosen_probs, color=plt.cm.RdYlGn(chosen_probs), edgecolor='black', linewidth=0.5)
    bars = axes[0].bar(steps, chosen_probs, color = 'blue', edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel("Prob(chosen token)")
    axes[0].set_title("Confidence of Chosen Next Token at Each Step")
    axes[0].set_ylim(0, 1.05)
    for i, (bar, token) in enumerate(zip(bars, chosen_tokens)):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     token, ha='center', va='bottom', fontsize=10, rotation=90)
    
    # Plot Entropy per step
    axes[1].plot(steps, entropies, marker='o', color='purple', linewidth=2)
    axes[1].fill_between(steps, entropies, alpha=0.2, color='purple')
    axes[1].set_xlabel("Generation Step")
    axes[1].set_ylabel("Entropy")
    axes[1].set_title("Prediction Entropy per Step")
    
    # plt.tight_layout()
    plt.savefig(f"viz/confidence_curves/generation_confidence_src_{source_layer}_tgt_{target_layer}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to generation_confidence.png")


def run_patchscopes_with_params(model, tokenizer, soft_prompt, target_prompt, num_tokens,
                                source_layer, target_layer, end_token=None):
    """
    Executes the Patchscopes technique for a given Soft Prompt and Target Prompt to interpret the
    Soft Prompt.
    
    :param model: Pre-trained LLM
    :param tokenizer: Tokenizer for the LLM
    :param soft_prompt: Learned Soft Prompt to inject (embedding vector)
    :param target_prompt: The prompt where we patch hidden states (eg: "The meaning of x x x is:")
    :param num_tokens: Number of tokens in the soft prompt
    :param source_layer: Which layer's hidden states to extract
    :param target_layer: Which layer to inject hidden states into
    :param end_token: Stopping token for generation
    """
    # TODO: Handle case when source prompt or target prompt is 0


    # Run soft prompt through the model and capture hidden states at every layer
    hs_cache, _ = build_soft_hs_cache(soft_prompt, model, tokenizer, num_tokens)

    # Tokenize the target prompt
    target_inp = tokenizer(target_prompt, return_tensors="pt").to(DEVICE)

    # Calculate position for patching (at placeholder token (x) positions)
    # target_position = target_inp["input_ids"].shape[1] - num_tokens # Can add -1 for the ":" in the end

    # Use target_position as 1, since first token is BOS
    target_position = 1

    # Set the stopping token for generation (defaults to comma)
    # NOTE: Bug: This is actually getting the BOS token as of now
    # end_token = end_token or tokenizer.encode(',')[0]

    # TODO: Try this
    end_token = end_token or tokenizer.encode(',', add_special_tokens=False)[0]

    # run model on the same prompt

    # Create a deep copy of target inp tensors
    target_inp_copy = {}
    for _k, _v in target_inp.items():
        target_inp_copy[_k] = _v.detach().clone().to(DEVICE)

    # Create hs_patch_config
    # hs_cache[i] stores hidden_states[i] which is:
    #   - i=0: embedding output (before any transformer layer)
    #   - i=1 to 32: output of layer i (or i - 1 according to model.model.layers indexing style)
    # So to get output of source_layer, we access hs_cache[source_layer + 1]
    hs_patch_config = {
        target_layer: [
            (target_position, hs_cache[source_layer + 1][0])
        ]
    }

    # Generate text with the patched hidden states
    return generate_greedy_deterministic(hs_patch_config, 
                                         target_inp_copy, 
                                         60, 
                                         end_token, 
                                         model, 
                                         tokenizer, 
                                         num_tokens, 
                                         target_layer = target_layer,
                                         source_layer = source_layer,
                                         do_sample = True,
                                         visualize_confidence = True)

