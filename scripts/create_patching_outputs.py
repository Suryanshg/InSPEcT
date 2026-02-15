import argparse
import os
from src.patchscopes.layers import get_layers_combinations_for_model
from src.patchscopes.patch_layers import run_patching_and_save
from src.patchscopes.target_prompt import few_shot_demonstrations, create_few_shot_prompt, create_cot_prompt
from src.utils import get_model, get_tokenizer
from datasets import load_dataset


if __name__ == '__main__':
    # Define All CLI Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The model name', required=True, nargs='?')
    parser.add_argument('-n', '--num_tokens', type=int, help='number of patched tokens', required=True, nargs='?')
    parser.add_argument('-d', '--dataset', type=str, help='the dataset name', required=True, nargs='?')
    parser.add_argument('-c', '--checkpoints_path', type=str, help='directory with soft prompts to run patching on', required=True, nargs='?')
    parser.add_argument('-tp', '--target_prompt', type=str, help='the target prompt', required=False, nargs='?')
    parser.add_argument('-tn', '--target_prompt_name', type=str, help='the target prompt name for results output', required=False, nargs='?')
    parser.add_argument('-t', '--target_prompt_type', type=str, help='the target prompt type to sample', required=True, nargs='?')
    parser.add_argument('-min', '--min_epoch', type=int, help='the epoch to start patching from', required=False, nargs='?')
    parser.add_argument('-max', '--max_epoch', type=int, help='the maximal epoch to patch', required=False, nargs='?')
    parser.add_argument('-j', '--jumps', type=int, help='the jump between epochs patchings', required=False, nargs='?')
    parser.add_argument('-i', '--index', type=int, help='sampled prompt index (an integer)', required=True, nargs='?')
    parser.add_argument('-f', '--first_epochs', type=int, help='how many first epochs to evaluate on without jumping', required=False, nargs='?')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory for results', required=False, nargs='?')

    # Parse all args
    args = parser.parse_args()
    model_path = args.model                                                 # Model Path
    num_tokens = args.num_tokens                                            # Number of Soft (Continuous) Prompt Tokens
    target_prompt_type = args.target_prompt_type                            # Either: 'classes_to_description' | 'description_and_classes'
    checkpoints_dir = args.checkpoints_path                                 # Dir where trained soft prompts are located
    task_dataset = args.dataset                                             # Dataset Path (For naming purposes)
    min_epoch = args.min_epoch or 0                                         # Min epoch to use for checkpointed soft prompt
    max_epoch = args.max_epoch or len(os.listdir(checkpoints_dir)) - 1      # Max epoch to use for checkpointed soft prompt
    jumps = args.jumps or 1                                                 # Jumps b/w each epoch (TODO Improve this)
    target_prompt_index = args.index or 3                                   # For Naming the Target Prompt using the Index value.
    first_epochs = args.first_epochs or 20                                  # Num of First epochs to use before jumping
    output_dir = args.output_dir                                            # Output dir for patched results

    # Retrieve examples to generate a target prompt (using Few Shot prompts)
    examples = few_shot_demonstrations.get(target_prompt_type)

    # If target prompt is provided by the user
    if args.target_prompt is not None:
        target_name = args.target_prompt_name or 'custom'
        target_prompts = {f"target_{target_name}_{target_prompt_index}": args.target_prompt}
    
    # Target prompt not provided by the user (Most likely to be used)
    else:
        target_name = target_prompt_type

        # Generate target prompts (a single one) using Few Shot examples
        # target_prompts = {f"target_{target_name}_{target_prompt_index}": create_few_shot_prompt(num_tokens, examples) for i in range(1)}


        # Fetch a random example from the test dataset
        # TODO: Can add seed to shuffle() later for reproducibility
        random_test_example = load_dataset(task_dataset, trust_remote_code=True, split='test').shuffle()[0]['sentence']

        # Generate chain-of-thought styled target prompt using random_test_example
        target_prompts = {f"target_{target_name}_{target_prompt_index}": create_cot_prompt(num_tokens, random_test_example) for i in range(1)}


    # Print the Target Prompt to be used for patching
    print("Using following Target Prompt for Patching:")
    print(target_prompts[f'target_{target_name}_{target_prompt_index}'])

    # Get Tokenizer, Model (LLM), and Layer Combinations for Patching Experiments
    tokenizer = get_tokenizer(model_path)
    model = get_model(model_path, to_device=True)
    layers_combinations = get_layers_combinations_for_model(model_path)

    # For all epochs in the checkpoints dir
    for f in os.listdir(checkpoints_dir):

        # Extract the epoch number
        epoch = int(f.split('_')[1])

        # If the extracted epoch is
        # 1. Less than min epoch (default is 0) OR
        # 2. More than max epoch (default is num of files in checkpoints dir - 1) OR 
        # 3. Diff b/w extracted and min epoch is not divisible by jumps (default jumps is 1) AND 
        #    extracted epoch is greater than first number of epochs to do before jumps (20 by default)
        # THEN skip to the next epoch
        if (epoch < min_epoch) or \
            (epoch > max_epoch) or \
            (((epoch - min_epoch) % jumps != 0) and (epoch > first_epochs)):
            continue

        # Construct path for the soft prompt file
        soft_prompt_path = f'{checkpoints_dir}/{f}'

        # For each target prompt (most likely always 1)
        for name, target_prompt in target_prompts.items():

            # Run Patching experiment and save the output
            run_patching_and_save(
                model_path, 
                name, 
                soft_prompt_path, 
                num_tokens, 
                task_dataset, 
                model, 
                tokenizer, 
                target_prompt,
                layer_combinations=layers_combinations,
                output_dir=output_dir,
            )
