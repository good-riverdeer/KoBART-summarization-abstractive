# KoBART-summarization (Abstractive)

KoBART abstractive summarization with **pytorch**

---

## Data

- [AI-Hub](https://aihub.or.kr/)
  - "[문서요약 텍스트](https://aihub.or.kr/aidata/8054)" 데이터 활용
    - Train: 325,072
    - Validation: 40,134
  - "[논문요약 텍스트](https://aihub.or.kr/aidata/30712)" 데이터 활용
    - Train: 715,652
    - Validation: 73,061

---

## Data unzip

```
python preprocess.py --data_root {directory/your/zipfile/is}
```

- `논문요약 텍스트/*.zip` -> `training.tsv`, `validation.tsv`

---

## Requirements

- transformers
- boto3
- rouge
- konlpy
- tqdm
- wandb

```
pip install -r requirements.txt
```

- install **pytorch** at https://pytorch.org/

---

## Training

```
python train.py --log --data_root dir/your/data/is --mecab_path dir/your/mecab/is
```

---

## References

- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization-pytorch](https://github.com/BM-K/KoBART-summarization-pytorch)
