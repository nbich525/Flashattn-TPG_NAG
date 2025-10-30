import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class InteriorDataset(Dataset):
    def __init__(self, json_path, tokenizer, size=512):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]
        image = Image.open(sample["image"]).convert("RGB").resize((self.size, self.size))
        prompt = sample["prompt"]
        inputs = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        return {
            "pixel_values": (torch.tensor(np.array(image)).permute(2,0,1)/255.0)*2 - 1,
            "input_ids": inputs.input_ids.squeeze(0)
        }
