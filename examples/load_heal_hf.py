from datasets import load_dataset

CONFIGS = [
    "distractor_injection",
    "object_removal",
    "scene_object_synonymous",
    "scene_task_contradiction",
]

for cfg in CONFIGS:
    try:
        ds = load_dataset("Trishna13/HEAL", cfg)
        print("\nCONFIG:", cfg)
        print(ds)
        for split in ds:
            print(split, ds[split][0] if len(ds[split]) else "EMPTY")
            break
    except Exception as e:
        print(f"{cfg}: {type(e).__name__}: {e}")
