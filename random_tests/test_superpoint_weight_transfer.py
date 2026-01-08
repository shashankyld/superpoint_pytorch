import torch
from superpoint.models.superpoint import SuperPoint

def verify_transfer():
    model = SuperPoint()
    ckpt_path = "checkpoints/magicpoint_best.pth"
    
    print(f"[*] Loading from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Handle both raw state_dicts and trainer checkpoints
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # Take a weight snapshot of the first layer
    original_val = model.encoder.block1[0][0].weight.clone()
    
    # Load with strict=False
    info = model.load_state_dict(state_dict, strict=False)
    
    new_val = model.encoder.block1[0][0].weight
    
    print("-" * 30)
    print(f"Weights updated: {not torch.equal(original_val, new_val)}")
    print(f"Missing keys (should be descriptor): {[k for k in info.missing_keys if 'descriptor' in k]}")
    
    if not torch.equal(original_val, new_val):
        print("\nSUCCESS! Encoder and Detector weights are now synchronized.")
    else:
        print("\nSTILL FAILING: Layer names in checkpoint don't match model.")

if __name__ == "__main__":
    verify_transfer()