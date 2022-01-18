import torch
import torch.nn as nn
from kobart import get_pytorch_kobart_model
from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast, BartModel


class KoBARTSummarization(nn.Module):
    def __init__(self, args):
        super(KoBARTSummarization, self).__init__()
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        self.kobart = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1',
                                                                   output_attentions=True,
                                                                   output_hidden_states=True)
        self.vocab_size = self.kobart.config.vocab_size

        self.args = args
        self.linear_copy = nn.Linear(self.kobart.config.hidden_size, 1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, is_test=False):
        if not is_test:
            outs = self.kobart(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'],
                               decoder_input_ids=inputs['decoder_input_ids'],
                               decoder_attention_mask=inputs['decoder_attention_mask'],
                               labels=inputs['labels'],
                               return_dict=True)
            encoder_input_ids = inputs['input_ids']

            logits = outs.logits
            last_hidden_state = outs.decoder_hidden_states[-1]
            last_attention_weight = torch.softmax(outs.cross_attentions[-1], dim=-1)

            p_copy = torch.sigmoid(self.linear_copy(last_hidden_state))
            previous_word_pro = torch.softmax(logits, dim=-1) * (1 - p_copy)

            encoder_word_attention = p_copy * torch.mean(last_attention_weight, dim=1)

            mask = torch.where(encoder_input_ids == self.tokenizer.pad_token_id,
                               encoder_word_attention.new_zeros(encoder_input_ids.shape),
                               encoder_word_attention.new_ones(encoder_input_ids.shape))

            encoder_word_attention = encoder_word_attention * mask.unsqueeze(1)
            personal_words = encoder_input_ids.unsqueeze(1).repeat(1, encoder_word_attention.shape[1], 1)
            word_pro = torch.scatter_add(previous_word_pro, 2, personal_words.long(), encoder_word_attention)

            loss = self.loss_fn(word_pro.view(-1, self.vocab_size), inputs['labels'].view(-1))

            return loss

        else:
            outputs = self.kobart.generate(inputs['input_ids'],
                                           max_length=self.args.max_len,
                                           num_beams=5)

            return outputs


class KoBartConditionalGeneration(nn.Module):
    def __init__(self, args, tokenizer):
        super(KoBartConditionalGeneration, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model(),
                                                                  output_attentions=True,
                                                                  output_hidden_states=True)
        self.vocab_size = self.model.config.vocab_size

        self.args = args
        self.linear_copy = nn.Linear(768, 1)
        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, is_test=False):
        if not is_test:
            outs = self.model(input_ids=inputs['input_ids'],
                              attention_mask=inputs['attention_mask'],
                              decoder_input_ids=inputs['decoder_input_ids'],
                              decoder_attention_mask=inputs['decoder_attention_mask'],
                              labels=inputs['labels'],
                              return_dict=True)
            encoder_input_ids = inputs['input_ids']

            logits = outs.logits
            last_hidden_state = outs.decoder_hidden_states[-1]
            last_attention_weight = torch.softmax(outs.cross_attentions[-1], dim=-1)

            p_copy = torch.sigmoid(self.linear_copy(last_hidden_state))
            previous_word_pro = torch.softmax(logits, dim=-1) * (1 - p_copy)

            encoder_word_attention = p_copy * torch.mean(last_attention_weight, dim=1)

            mask = torch.where(encoder_input_ids == self.tokenizer.pad_token_id,
                               encoder_word_attention.new_zeros(encoder_input_ids.shape),
                               encoder_word_attention.new_ones(encoder_input_ids.shape))

            encoder_word_attention = encoder_word_attention * mask.unsqueeze(1)
            personal_words = encoder_input_ids.unsqueeze(1).repeat(1, encoder_word_attention.shape[1], 1)
            word_pro = torch.scatter_add(previous_word_pro, 2, personal_words.long(), encoder_word_attention)

            loss = self.loss_fn(word_pro.view(-1, self.vocab_size), inputs['labels'].view(-1))

            return loss

        else:
            outputs = self.model.generate(inputs['input_ids'],
                                          max_length=self.args.max_len,
                                          num_beams=5)

            return outputs
