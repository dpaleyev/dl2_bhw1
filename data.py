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
    if not os.path.exists(config.TXT_DIR):
        json_to_txt()
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
    def __init__(self, dataset_dir, txt_file, tokenizer, train=True):
        self.tokenizer = tokenizer
        self.data_files = []
        self.dataset_dir = dataset_dir
        self.train = train
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir , exist_ok=True)
            self.load_data_from_txt_file(txt_file)
        else:
            for filename in sorted(os.listdir(self.dataset_dir)):
                if filename.endswith('.npy'):
                    self.data_files.append(filename)


    def load_data_from_txt_file(self, file):
        with open(file) as file_txt:
            for idx, line in tqdm(enumerate(file_txt), f"Loading {'train' if self.train else 'val'} data"):
                tokens = self.tokenizer.encode(line)
                npy_filename = f"data_{idx}.npy"
                np.save(os.path.join(self.dataset_dir, npy_filename), tokens)
                self.data_files.append(npy_filename)
        

    def load_data_from_json(self):
        for filename in tqdm(os.listdir(config.JSON_DIR), "Loading data"):
            if not filename.endswith('.json'):
                continue
            with open(os.path.join(config.JSON_DIR, filename), 'r') as file_json:
                data = json.load(file_json)
                for idx, item in enumerate(data):
                    tokens = self.tokenizer.encode(item['story'])
                    npy_filename = f"{filename}_{idx}.npy"
                    np.save(os.path.join(self.dataset_dir, npy_filename), tokens)
                    self.data_files.append(npy_filename)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        BOS_ID = self.tokenizer.bos_id()
        EOS_ID = self.tokenizer.eos_id()
        PAD_ID = self.tokenizer.pad_id()

        tokens = list(np.load(os.path.join(self.dataset_dir, self.data_files[idx])))
        indices = [BOS_ID] + tokens[:config.MAX_SEQ_LEN - 2] + [EOS_ID]
        length = len(indices)
        pad = torch.full((config.MAX_SEQ_LEN,), PAD_ID, dtype=torch.int64)
        pad[:length] = torch.tensor(indices)
        padded_seq =  torch.tensor(pad)

        padding_mask = (padded_seq == PAD_ID)

        return padded_seq, padding_mask