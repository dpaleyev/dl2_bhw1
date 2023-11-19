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
        tokens = np.load(os.path.join(config.DATASET_DIR, self.data_files[idx]))
        return tokens

    def get_collator(self):
        BOS_ID = self.tokenizer.bos_id()
        EOS_ID = self.tokenizer.eos_id()
        PAD_ID = self.tokenizer.pad_id()

        def collate_fn(batch):
            padded_batch = [torch.tensor([BOS_ID] + item + [EOS_ID]) for item in batch]
            padded_batch = torch.nn.utils.rnn.pad_sequence(padded_batch, padding_value=PAD_ID, batch_first=True)
            padding_mask = (padded_batch == PAD_ID)
            return {
                "tokens": padded_batch,
                "padding_mask": padding_mask,
            }
        
        return collate_fn