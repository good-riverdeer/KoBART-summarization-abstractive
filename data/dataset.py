import os
import csv
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset


class SummarizationData(Dataset):
    def __init__(self, args, split, tokenizer):
        super(SummarizationData, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        file_path = os.path.join(args.data_root, f"{split}.tsv")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        self.eos_token_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
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

        input_ids = self.pad_sequence(self.tokenizer.encode(origin), self.pad_token_idx)
        label_ids = self.tokenizer.encode(summary) + [self.eos_token_idx]
        dec_input_ids = self.pad_sequence([self.pad_token_idx] + label_ids[:-1], self.pad_token_idx)
        label_ids = self.pad_sequence(label_ids, self.ignore_index)

        input_ids = torch.tensor(input_ids).to(self.args.device)
        dec_input_ids = torch.tensor(dec_input_ids).to(self.args.device)
        label_ids = torch.tensor(label_ids).long().to(self.args.device)

        attention_mask = input_ids.ne(self.pad_token_idx).float()
        decoder_attention_mask = dec_input_ids.ne(self.pad_token_idx).float()

        ret = {'input_ids': input_ids,
               'attention_mask': attention_mask,
               'decoder_input_ids': dec_input_ids,
               'decoder_attention_mask': decoder_attention_mask,
               'labels': label_ids}
        return ret


class SummarizationDataset(Dataset):
    def __init__(self, args, split, tokenizer):
        assert split == 'training' or 'validation'
        super(SummarizationDataset, self).__init__()
        self.args = args

        self.tokenizer = tokenizer
        self.file_path = os.path.join(args.data_root, f"{split}.tsv")

        self.input_ids = []
        self.attention_mask = []
        self.decoder_input_ids = []
        self.decoder_attention_mask = []
        self.labels = []

        self.init_token = self.tokenizer.bos_token      # <s>
        self.pad_token = self.tokenizer.pad_token       # <pad>
        self.unk_token = self.tokenizer.unk_token       # <unk>
        self.eos_token = self.tokenizer.eos_token       # </s>

        self.init_token_idx = self.tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        self.eos_token_idx = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.ignore_index = -100

        self.load_data()

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # lines = csv.reader(f, delimiter='\t')
            lines = f.readlines()

            with tqdm(lines) as iteration:
                iteration.set_description("loading data")
                for line in iteration:
                    # origin, summary = line
                    origin, summary = line.split('\t')
                    origin = origin.strip()
                    summary = summary.strip()

                    input_ids = self.pad_sequence(self.tokenizer.encode(origin), self.pad_token_idx)
                    label_ids = self.tokenizer.encode(summary) + [self.eos_token_idx]
                    dec_input_ids = self.pad_sequence([self.pad_token_idx] + label_ids[:-1], self.pad_token_idx)
                    label_ids = self.pad_sequence(label_ids, self.ignore_index)

                    input_ids = torch.tensor(input_ids)
                    dec_input_ids = torch.tensor(dec_input_ids)
                    label_ids = torch.tensor(label_ids)

                    attention_mask = input_ids.ne(self.pad_token_idx).float()
                    decoder_attention_mask = dec_input_ids.ne(self.pad_token_idx).float()

                    self.input_ids.append(input_ids)
                    self.attention_mask.append(attention_mask)
                    self.decoder_input_ids.append(dec_input_ids)
                    self.decoder_attention_mask.append(decoder_attention_mask)
                    self.labels.append(label_ids)

    def pad_sequence(self, inputs, padding_token_idx):
        if len(inputs) < self.args.max_len:
            pad = np.array([padding_token_idx] * (self.args.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.args.max_len]
        return inputs

    def __getitem__(self, idx):
        device = self.args.device
        ret = {'input_ids': self.input_ids[idx].to(device),
               'attention_mask': self.attention_mask[idx].to(device),
               'decoder_input_ids': self.decoder_input_ids[idx].to(device),
               'decoder_attention_mask': self.decoder_attention_mask[idx].to(device),
               'labels': self.labels[idx].long().to(device)}
        return ret

    def __len__(self):
        return len(self.labels)
