"""Quick probe — run: python probe_dataset.py"""
from datasets import load_dataset, get_dataset_config_names

print("Available configs:")
try:
    configs = get_dataset_config_names("McAuley-Lab/Amazon-C4")
    print(configs)
except Exception as e:
    print(f"  Could not fetch configs: {e}")

print("\nLoading first 3 items (streaming, split=test) …")
try:
    ds = load_dataset("McAuley-Lab/Amazon-C4", split="test", streaming=True)
    for i, item in enumerate(ds):
        if i >= 3:
            break
        print(f"\n--- item {i} ---")
        print(f"  keys : {list(item.keys())}")
        for k, v in item.items():
            print(f"  {k!r}: {str(v)[:120]}")
except Exception as e:
    print(f"Error: {e}")
