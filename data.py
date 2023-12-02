import sentencepiece as spm
from tqdm import tqdm
import os
import json
import glob
import torch
import numpy as np

import config


def json_to_txt():
    os.makedirs(config.TXT_DIR, exist_ok=True)

    for filename in tqdm(os.listdir(config.JSON_DIR), "Copying json to txt"):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(config.JSON_DIR, filename), 'r') as file_json:
            data = json.load(file_json)
            with open(os.path.join(config.TXT_DIR, filename.replace('.json', '.txt')), 'w') as file_txt:
                file_txt.write("\n".join(item["story"] for item in data))

def get_tokenizer():
    if not os.path.exists(config.TOKENIZER_PATH):
        txt_files = glob.glob(f"{config.TXT_DIR}/*.txt")
        txt_files_str = ','.join(txt_files)
        spm.SentencePieceTrainer.train(
            input=txt_files_str,
            vocab_size=config.VOCAB_SIZE,
            model_prefix=config.TOKENIZER_PATH.replace('.model', ''),
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
    return spm.SentencePieceProcessor(model_file=config.TOKENIZER_PATH)


class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_files = []
        if not os.path.exists(config.DATASET_DIR):
            os.makedirs(config.DATASET_DIR, exist_ok=True)
            self.load_data()

    def load_data(self):
        for filename in tqdm(os.listdir(config.JSON_DIR)[:2], "Loading data"):
            if not filename.endswith('.json'):
                continue
            with open(os.path.join(config.JSON_DIR, filename), 'r') as file_json:
                data = json.load(file_json)
                for idx, item in enumerate(data):
                    tokens = self.tokenizer.encode(item['story'])
                    npy_filename = f"{filename}_{idx}.npy"
                    np.save(os.path.join(config.DATASET_DIR, npy_filename), tokens)
                    self.data_files.append(npy_filename)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        BOS_ID = self.tokenizer.bos_id()
        EOS_ID = self.tokenizer.eos_id()
        PAD_ID = self.tokenizer.pad_id()

        tokens = np.load(os.path.join(config.DATASET_DIR, self.data_files[idx]))
        indices = [BOS_ID] + tokens[:config.MAX_SEQ_LEN - 2] + [EOS_ID]
        length = len(indices)
        pad = torch.full((config.MAX_SEQ_LEN,), PAD_ID, dtype=torch.int64)
        pad[:length] = torch.tensor(indices)

        return torch.tensor(pad)