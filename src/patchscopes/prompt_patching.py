import torch
from src.constants import DEVICE


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
    
    # Raise Error if model is any other than a Llama
    else:
        raise ValueError(f"Unknown model: {model.config.model_type}")
    
    # Attach the hook to layer 0 (so soft_prompt enters at the start of processing)
    hooks = [first_layer.register_forward_hook(
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
    
    # Inject soft_prompt at layer 0
    patch_hooks = set_soft_prompt_patch_hook(model, soft_prompt, pos, num_of_tokens) 

    # Run forward pass of the placeholder tokens
    with torch.no_grad():
        output = model(**inp, output_hidden_states = True)

    # Remove all hooks (call-back methods after each forward call on a layer)
    remove_hooks(patch_hooks)

    # For each layer idx in layers to cache
    for layer in layers_to_cache:

        # If layer is not in cache yet
        if layer not in hs_cache:

            # Allocate an empty list for that layer
            hs_cache[layer] = []

        # Store hidden state at the patched position
        hs_cache[layer].append(output["hidden_states"][layer][0][pos:])

    # Return the hidden state cache and the tokenized placeholder text
    return hs_cache, inp


def generate_greedy_deterministic(hs_patch_config, 
                                  inp, 
                                  max_length, 
                                  end_token, 
                                  model, 
                                  tokenizer, 
                                  num_of_tokens, 
                                  target_layer,
                                  do_sample = False,
                                  temperature = 1.0,
                                  top_p = 0.9,
                                  top_k = 50):
    
    # Copy target prompt's input token ids for generation (IDK why again)
    input_ids = inp["input_ids"].detach().clone().to(DEVICE)

    # TODO: Try this to see if it works
    model.set_attn_implementation('eager')

    with torch.no_grad():
        for _ in range(max_length):
            patch_hooks = set_hs_patch_hooks(model, hs_patch_config, num_of_tokens) 
            outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
            remove_hooks(patch_hooks)
            
            # Extract logits for the last token
            logits = outputs.logits[:, -1, :]

            # Get the next token id using the logits
            if do_sample:
                # Apply temperature scaling
                logits = logits / temperature

                # Sample from the softmax distribution
                probs = torch.softmax(logits, dim = -1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)

            else:
                next_token_id = torch.argmax(logits, dim=-1)

            # Concat the next token id to the input_ids for autoregressive generation
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            # If the end_token is predicted, break out of the autoregression loop
            if next_token_id.item() == end_token:
                break

    generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    patched_pattern = (" x" * num_of_tokens).strip()
    return "".join(generated_text).split(patched_pattern)[-1] 


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
    # Run soft prompt through the model and capture hidden states at every layer
    hs_cache, _ = build_soft_hs_cache(soft_prompt, model, tokenizer, num_tokens)

    # Tokenize the target prompt
    target_inp = tokenizer(target_prompt, return_tensors="pt").to(DEVICE)

    # Calculate position for patching (at placeholder token (x) positions)
    target_position = target_inp["input_ids"].shape[1] - num_tokens # Can add -1 for the ":" in the end

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
    #   - i=1 to 32: output of layer (i-1)
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
                                         target_layer,
                                         do_sample=True)

