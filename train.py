import torch
from torch.utils.data import DataLoader

import os
import datetime
import wandb
from tqdm import tqdm
from argparse import ArgumentParser

from transformers import get_linear_schedule_with_warmup
from kobart import get_kobart_tokenizer

from data.dataset import SummarizationDataset
from modeling.metric import Metric
from modeling.model import KoBartConditionalGeneration


class Trainer:
    def __init__(self, args):
        self.args = args
        self.tokenizer = get_kobart_tokenizer()

        train_set = SummarizationDataset(os.path.join(args.data_root, args.train_file),
                                         args=args,
                                         tokenizer=self.tokenizer)
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        valid_set = SummarizationDataset(os.path.join(args.data_root, args.valid_file),
                                         args=args,
                                         tokenizer=self.tokenizer)
        self.valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

        self.model = KoBartConditionalGeneration(args, self.tokenizer).to(self.args.device)

        self.metric = Metric(args)
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

        train_total = len(self.train_loader) * self.args.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * train_total,
                                                         num_training_steps=train_total)

        self.checker = {'early_stop_patient': 0,
                        'best_valid_loss': float('inf')}
        self.progress = {'loss': -1, 'iter': -1}

        if args.log:
            model_name = datetime.datetime.now().strftime("%m%d-%H%M")
            wandb.init(project='KoBART-summarization-abstractive', config=self.args, name=model_name)

    def train(self):
        self.model.train()
        self.progress = self.progress.fromkeys(self.progress, 0)

        with tqdm(self.train_loader, unit='batch') as iteration:
            for step, batch in enumerate(iteration):
                iteration.set_description("{:10s}".format("Train"))
                self.optimizer.zero_grad()

                loss = self.model(batch)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                self.progress['loss'] += loss.cpu().detach().numpy()
                self.progress['iter'] += 1

                avg_loss = self.progress['loss'] / self.progress['iter']
                iteration.set_postfix_str(f"loss - {avg_loss:.4f}")

        return avg_loss

    def eval(self):
        self.model.eval()
        self.progress = self.progress.fromkeys(self.progress, 0)

        with torch.no_grad():
            with tqdm(self.train_loader, unit='batch') as iteration:
                for step, batch in enumerate(iteration):
                    iteration.set_description("{:10s}".format("Train"))

                    loss = self.model(batch)

                    self.progress['loss'] += loss.cpu().detach().numpy()
                    self.progress['iter'] += 1
                    self.metric.generation(self.model, self.tokenizer, batch)

                    avg_loss = self.progress['loss'] / self.progress['iter']
                    avg_rouge = self.metric.avg_rouge()

                    postfix = f"loss - {avg_loss:.4f}, rouge-1[f_score] - {avg_rouge['rouge-1']['f']:.4f}, "
                    postfix += f"rouge-2[f_score] - {avg_rouge['rouge-2']['f']:.4f}, "
                    postfix += f"rouge-l[f_score] - {avg_rouge['rouge-l']['f']:.4f}"
                    iteration.set_postfix_str(postfix)

        return avg_loss, avg_rouge

    def run(self):
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train()
            valid_loss, valid_rouge = self.eval()

            # early stop
            if valid_loss < self.checker['best_valid_loss']:
                self.checker['early_stop_patient'] = 0
                self.checker['best_valid_loss'] = valid_loss
                if self.args.log:
                    torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, 'best_model.pt'))
                    print(f"SAVE Model - validation loss: {valid_loss: .4f}")
            else:
                self.checker['early_stop_patient'] += 1
                if self.checker['early_stop_patient'] == self.args.patient:
                    print(f'Early stopping - epoch: {epoch}')
                    break

            # wandb log
            if self.args.log:
                log_dct = {'train loss': train_loss,
                           'validation loss': valid_loss,
                           'rouge-1[recall]': valid_rouge['rouge-1']['r'],
                           'rouge-1[precision]': valid_rouge['rouge-1']['p'],
                           'rouge-1[f_score]': valid_rouge['rouge-1']['f'],
                           'rouge-2[recall]': valid_rouge['rouge-2']['r'],
                           'rouge-2[precision]': valid_rouge['rouge-2']['p'],
                           'rouge-2[f_score]': valid_rouge['rouge-2']['f'],
                           'rouge-l[recall]': valid_rouge['rouge-l']['r'],
                           'rouge-l[precision]': valid_rouge['rouge-l']['p'],
                           'rouge-l[f_score]': valid_rouge['rouge-l']['f']}
                wandb.log(log_dct)


def print_args(arguments):
    for idx, (k, v) in enumerate(arguments.__dict__.items()):
        if idx == 0:
            print("Arguments {\n", "\t", k, ":", v)
        elif idx == len(arguments.__dict__) - 1:
            print("\t", k, ":", v, "\n}")
        else:
            print("\t", k, ":", v)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--train_file', type=str, default='training.tsv')
    parser.add_argument('--valid_file', type=str, default='validation.tsv')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--mecab_path', type=str, default=None)

    parser.add_argument('--patient', type=int, default=5)
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_args(args)

    trainer = Trainer(args)
    trainer.run()
