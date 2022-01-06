import os
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

    # for zip_file in glob(os.path.join(zip_file_root, '*', split, '*.zip')):
    for zip_file in glob(os.path.join(zip_file_root, split, '*.zip')):

        zf = zipfile.ZipFile(zip_file)
        for json_file_name in zf.namelist():
            json_file = zf.read(json_file_name)
            print(f'unzip {os.path.basename(zip_file)}')

            if '문서요약 텍스트' in zip_file:
                dicts = json.loads(json_file)['documents']
                for d in tqdm(dicts):
                    origin, summary = document_summarization(d)
                    writer.writerow([origin, summary])

            # elif '논문자료 요약':
            #     dicts = json.loads(json_file)['data']
            #     print()

    f.close()


def document_summarization(dct):
    sentences = list(itertools.chain(*dct['text']))
    origin_text = ' '.join([item['sentence'] for item in sentences])
    summary_text = ' '.join([item['sentence'] for item in sentences if item['index'] in dct['extractive']])
    return origin_text, summary_text


# def paper_summarization(dct):


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
