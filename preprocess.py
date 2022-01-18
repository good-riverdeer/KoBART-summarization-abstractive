import os
import re
import csv
import zipfile
import itertools
import json
from argparse import ArgumentParser

from tqdm import tqdm
from glob import glob


def unzip(zip_file_root, split='Validation'):
    f = open(os.path.join(zip_file_root, split.lower() + '.tsv'), 'w', encoding='utf-8', newline='\n')
    writer = csv.writer(f, delimiter='\t')

    for zip_file in glob(os.path.join(zip_file_root, f'*/{split}/*.zip')):
        zf = zipfile.ZipFile(zip_file)
        for json_file_name in zf.namelist():
            json_file = zf.read(json_file_name)
            print(f'unzip {json_file_name} from {zip_file}')

            if '문서요약 텍스트' in zip_file:
                dicts = json.loads(json_file)['documents']
                for d in tqdm(dicts):
                    origin, summary = document_summarization(d)
                    writer.writerow([origin, summary])

            elif '논문자료 요약' in zip_file:
                dicts = json.loads(json_file)['data']
                for d in tqdm(dicts):
                    sentences = paper_summarization(d)
                    if len(sentences) != 0:
                        writer.writerows(sentences)

    f.close()


def document_summarization(dct):
    sentences = list(itertools.chain(*dct['text']))
    origin_text = ' '.join([re.sub('\n|\t|\r', '', item['sentence']) for item in sentences])
    summary_text = ' '.join([re.sub('\n|\t|\r', '', item['sentence'])
                             for item in sentences if item['index'] in dct['extractive']])

    return origin_text, summary_text


def paper_summarization(dct):
    keys = list(filter(lambda x: 'summary' in x, dct.keys()))
    ret = []
    for key in keys:
        origin_text = dct[key][0]['orginal_text']
        summary_text = dct[key][0]['summary_text']
        origin_text = re.sub('\n|\t|\r', ' ', origin_text)
        summary_text = re.sub('\n|\t|\r', ' ', summary_text)
        if len(origin_text) != 0:
            ret.append([origin_text, summary_text])
    return ret


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--except_train', action='store_false')
    parser.add_argument('--except_validation', action='store_false')

    args = parser.parse_args()

    if args.except_validation:
        unzip(args.data_root, split='Validation')
    if args.except_train:
        unzip(args.data_root, split='Training')
