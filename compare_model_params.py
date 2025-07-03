import torch

def compare_models(file1, file2):
    # Load both model states on CPU
    state1 = torch.load(file1, map_location='cpu')
    state2 = torch.load(file2, map_location='cpu')
    
    # Get model parameters
    params1 = state1['module']
    params2 = state2['module']
    
    # Compare all parameters
    all_equal = True
    for (k1, v1), (k2, v2) in zip(params1.items(), params2.items()):
        if k1 != k2:
            print(f"Parameter name mismatch: {k1} vs {k2}")
            all_equal = False
            continue
        if not torch.equal(v1, v2):
            print(f"Parameter {k1} differs between models")
            all_equal = False
    
    if all_equal:
        print("All parameters are identical between the two models")
    else:
        print("Models have differences in parameters")

def compare_optimizers(file1, file2):
    # Load both optimizer states on CPU with weights_only=False for DeepSpeed support
    state1 = torch.load(file1, map_location='cpu', weights_only=False)
    state2 = torch.load(file2, map_location='cpu', weights_only=False)
    
    # Compare optimizer states
    if state1.keys() != state2.keys():
        print("Optimizer state keys differ")
        return False
    
    all_equal = True
    for key in state1:
        if isinstance(state1[key], torch.Tensor):
            if not torch.equal(state1[key], state2[key]):
                print(f"Optimizer state {key} differs")
                all_equal = False
        elif state1[key] != state2[key]:
            print(f"Optimizer state {key} differs (non-tensor)")
            all_equal = False
    
    if all_equal:
        print("All optimizer states are identical")
    else:
        print("Optimizer states have differences")
    return all_equal

if __name__ == "__main__":
    # Compare model parameters
    model1 = "workflow/train_text_sql/model_save/sql_lora/best/zero_pp_rank_0_mp_rank_00_model_states.pt"
    model2 = "workflow/train_text_sql/model_save/sql_lora/best/zero_pp_rank_1_mp_rank_00_model_states.pt"
    print("=== Comparing model parameters ===")
    compare_models(model1, model2)
    
    # Compare optimizer states
    optim1 = "workflow/train_text_sql/model_save/sql_lora/best/zero_pp_rank_0_mp_rank_00_optim_states.pt"
    optim2 = "workflow/train_text_sql/model_save/sql_lora/best/zero_pp_rank_1_mp_rank_00_optim_states.pt"
    print("\n=== Comparing optimizer states ===")
    compare_optimizers(optim1, optim2)
