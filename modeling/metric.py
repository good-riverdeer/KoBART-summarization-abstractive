import re

from rouge import Rouge
from konlpy.tag import Mecab


class Metric:
    def __init__(self, args):
        self.args = args
        self.step = 0
        self.rouge = Rouge()
        self.rouge_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
        self.mecab = Mecab(dicpath=args.mecab_path) if args.mecab_path is not None else Mecab()
        self.re_pattern = re.compile("[^A-Za-z0-9가-힣]")

        # model_name = datetime.datetime.now().strftime("%m%d-%H%M")
        # wandb.init(project='KoBART-summarization-abstractive', config=self.args, name=model_name)

    # def calculate_time(self, start_time, end_time):
    #     elapsed_time = end_time - start_time
    #     elapsed_mins = int(elapsed_time / 60)
    #     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    #     return elapsed_mins, elapsed_secs

    def avg_rouge(self):
        for metric, scores in self.rouge_scores.items():
            for k, v in scores.items():
                self.rouge_scores[metric][k] /= self.step
        return self.rouge_scores

    def generation(self, model, tokenizer, inp):
        output = model(inp, is_test=True)
        for step, beam in enumerate(output):
            ref = tokenizer.decode(inp['decoder_input_ids'][step], skip_special_tokens=True)
            hyp = tokenizer.decode(beam, skip_special_tokens=True)

            ref = ' '.join(self.mecab.morphs(self.re_pattern.sub(' ', ref.lower()).strip()))
            hyp = ' '.join(self.mecab.morphs(self.re_pattern.sub(' ', hyp.lower()).strip()))

            score = self.rouge.get_scores(hyp, ref)[0]
            for metric, scores in self.rouge_scores.items():
                for k, v in scores.items():
                    self.rouge_scores[metric][k] += score[metric][k]

            self.step += 1
