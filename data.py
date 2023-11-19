import sentencepiece as spm
from tqdm import tqdm
import os
import json
import glob

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
        



