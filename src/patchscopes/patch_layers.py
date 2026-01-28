import csv
import os
from tqdm import tqdm
from src.patchscopes.layers import all_small_layers
from src.patchscopes.prompt_patching import run_patchscopes_with_params
import torch
from src.utils import get_model, get_tokenizer

from src.constants import DEVICE


def run_patchscopes_for_layers_combinations(model, tokenizer, 
                                            soft_prompt, target_prompt, num_tokens,
                                            combinations=None, end_token=None, print_out=False):
    results = []
    combinations = combinations or all_small_layers

    # Calculate total iterations for progress bar
    total_iterations = sum(
        (comb.get("max_source") - comb.get("min_source") + 1) * 
        (comb.get("max_target") - comb.get("min_target") + 1)
        for comb in combinations
    )

    with tqdm(total=total_iterations, desc="Patchscopes experiments") as pbar:
        # For each combination (Default: There are like 6 of them)
        for comb in combinations:

            # For each source layer ranging from min to max value
            for source_layer in range(comb.get("min_source"), comb.get("max_source") + 1):

                # For each target layer ranging from min to max value
                for target_layer in range(comb.get("min_target"), comb.get("max_target") + 1):

                    # Run patchscopes experiment using the src and tgt layers
                    patched_output = run_patchscopes_with_params(
                        model, 
                        tokenizer, 
                        soft_prompt, 
                        target_prompt, 
                        num_tokens, 
                        source_layer, 
                        target_layer,
                        end_token
                    )

                    # Print out details after the experiment run
                    if print_out:
                        print(f"results for {source_layer=} {target_layer=}:")
                        print(f"{patched_output=}")
                        print()

                    # Append the experiment output to the results list
                    results.append({
                        "source_layer": source_layer,
                        "target_layer": target_layer,
                        "output": patched_output
                    })

                    pbar.update(1)

    # Return the results list comprising of patchscopes output
    # for each and every combination of source and target layers
    return results


def get_output_file_path(model_path, task_dataset, num_tokens, soft_prompt_path, 
                         target_prompt_name, output_dir=None):
    model = model_path.split('/')[-1]
    soft_prompt = soft_prompt_path.split('/')[-1][:-3]
    output_dir = output_dir or "./patching_output"
    output_path = f'{output_dir}/{model}/{task_dataset}/n{num_tokens}_{target_prompt_name}'
    os.makedirs(output_path, exist_ok=True)
    return f'{output_path}/{soft_prompt}.csv'


def write_results_to_csv(results, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["source_layer", "target_layer", "output"], escapechar='\\')
        writer.writeheader()
        writer.writerows(results)


def run_patching_and_save(model_path, target_prompt_name, soft_prompt_path, num_tokens, task_dataset, 
                          model=None, tokenizer=None, target_prompt=None, layer_combinations=None, output_dir=None):
    # Safely load tokenizer and model
    if tokenizer is None:
        tokenizer = get_tokenizer(model_path)

    if model is None:
        model = get_model(model_path, to_device=True)

    # Load the Trained Soft prompt and move it to device
    soft_prompt = torch.load(soft_prompt_path)
    soft_prompt = soft_prompt.to(DEVICE) 

    # TODO:?
    end_token = tokenizer.encode('$')[0]
    
    # Prepare a path to store patching experiment output
    # This will be a CSV file
    output_path = get_output_file_path(
        model_path, task_dataset, num_tokens, 
        soft_prompt_path, target_prompt_name, output_dir)
    
    # If the output path already exists, then return immediately
    if os.path.exists(output_path):
        return

    # Run the Patching experiments for each layer combination
    results = run_patchscopes_for_layers_combinations(
        model, 
        tokenizer, 
        soft_prompt, 
        target_prompt, 
        num_tokens,
        end_token=end_token,
        combinations=layer_combinations
    )
    
    write_results_to_csv(results, output_path)
    