# KoBART-summarization (Abstractive)

KoBART abstractive summarization with **pytorch**

---

## Data

- [AI-Hub](https://aihub.or.kr/)
  - "[문서요약 텍스트](https://aihub.or.kr/aidata/8054)" 데이터 활용
    - Train: 
    - Validation:
  - "[논문요약 텍스트](https://aihub.or.kr/aidata/30712)" 데이터 활용 (comming soon)
    - Train:
    - Validation:

---

## Data unzip

```
python preprocess.py --data_root directory/your/zipfile/is
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

---

## Training

```
python train.py --log --data_root dir/your/data/zipfile/is --mecab_path dir/your/mecab/is
```

---

## References

- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization-pytorch](https://github.com/BM-K/KoBART-summarization-pytorch)
