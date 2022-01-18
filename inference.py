import json
from argparse import ArgumentParser, Namespace

import torch

from modeling.model import KoBARTSummarization


class Inference:
    def __init__(self, args):
        with open(f"{args.model.replace('best_model.pt', 'config.json')}", 'r', encoding='utf-8') as f:
            model_args = json.load(f)
            model_args = Namespace(**model_args)

        self.args = args
        self.model = KoBARTSummarization(model_args).to(args.device)
        self.tokenizer = self.model.tokenizer
        self.model.load_state_dict(torch.load(args.model, map_location=args.device))

    def __call__(self, input_text):
        input_tokens = self.tokenizer.encode(input_text)
        tokens = torch.tensor(input_tokens).long().unsqueeze(0).to(self.args.device)

        pred = self.model({"input_ids": tokens}, is_test=True)
        out = self.tokenizer.decode(pred[0][1:-1])
        return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inf = Inference(args)
    pred = inf("본 연구의 목적은 국내 임상 검사실에서 대변 검사의 내부정도 관리 현황을 파악하기 위한 것이다. 대변 검경 검사를 시행하고 있는 국내 임상 검사실을 대상으로 하여 대변 검사의 정도관리 수행에 관한 전자우편 설문을 시행하였다. 설문에 응답한 총 39개 기관 중 20개 기관(51.3%)에서 대변 농축법을 통한 검사를 수행한다고 답변하였으며, 28개 기관(71.8%)에서 생리식염수법을 이용한 슬라이드 검경만 하고 있다고 답변하였다. 응답한 기관 중 대부분(74.4%)이 적절한 정도관리 물질을 확보하기 어려워 내부정도관리를 시행하지 못하고 있다고 응답하였다. 오직 4개 기관(10.3%)이 정기적으로 염색약의 반응도를 점검하고 있었다. 적절한 외부정도관리법으로 선호하는 방법으로는 정도관리 슬라이드의 배포(43.6%)가 가장 많았고, 다음으로 정도관리 물질 자체의 배포(30.8%)나 가상 슬라이드(17.9%), 또는 이들의 조합(7.7%) 순이었다. 국내 검사실에서 대변 검경 시 흔하게 관찰되는 기생충은 간흡충(75%), 왜소아메바, 요충, 대장아메바 순이었다. 본 연구를 통해 국내 검사실에서 대변 검경 검사의 내부정도관리가 어려운 것은 표준화된 정도관리 물질과 체계의 부재에서 기인함을 알 수 있었다. 본 연구 결과가 향후 대변 검경 검사의 적절한 정도관리 체계의 구축에 기반이 되리라 기대한다.")
    # 본 연구의 목적은 국내 임상 검사실에서 대변 검사의 내부정도 관리 현황을 파악하기 위한 것이다. 국내 임상 검사실을 대상으로 하여 대변 검사의 정도관리 수행에 관한 전자우편 설문을 시행하였다. 본 연구를 통해 국내 검사실에서 대변 검경 검사의 내부정도관리가 어려운 것은 표준화된 정도관리 물질과 체계의 부재에서 기인함을 알 수 있었다.
