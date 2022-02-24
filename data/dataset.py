import os
import csv
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    def __init__(self, args, split, tokenizer):
        super(SummarizationDataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        file_path = os.path.join(args.data_root, f"{split}.tsv")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        # self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        # self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        # self.eos_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.ignore_index = -100

    def pad_sequence(self, inputs, padding_token_idx):
        if len(inputs) < self.args.max_len:
            pad = np.array([padding_token_idx] * (self.args.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.args.max_len]
        return inputs

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        samples = self.lines[idx].split('\t')
        origin, summary = samples
        origin = origin.strip()
        summary = summary.strip()

        input_ids = self.tokenizer.encode(origin)
        input_ids = self.pad_sequence(input_ids, self.tokenizer.pad_token_id)
        label_ids = self.tokenizer.encode(summary) + [self.tokenizer.eos_token_id]

        # dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids = [self.tokenizer.eos_token_id] + label_ids[:-1]
        dec_input_ids = self.pad_sequence(dec_input_ids, self.tokenizer.pad_token_id)
        label_ids = self.pad_sequence(label_ids, self.ignore_index)

        input_ids = torch.tensor(input_ids).to(self.args.device)
        dec_input_ids = torch.tensor(dec_input_ids).to(self.args.device)
        label_ids = torch.tensor(label_ids).long().to(self.args.device)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).float()
        decoder_attention_mask = dec_input_ids.ne(self.tokenizer.pad_token_id).float()

        ret = {'input_ids': input_ids,
               'attention_mask': attention_mask,
               'decoder_input_ids': dec_input_ids,
               'decoder_attention_mask': decoder_attention_mask,
               'labels': label_ids}
        return ret
