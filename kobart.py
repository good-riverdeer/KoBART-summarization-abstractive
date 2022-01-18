# coding=utf-8
# Modified MIT License

# Software Copyright (c) 2020 SK telecom

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import shutil
import hashlib
from zipfile import ZipFile

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from transformers import PreTrainedTokenizerFast


class AwsS3Downloader(object):
    def __init__(
        self,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        self.resource = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        ).resource("s3")
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version=UNSIGNED),
        )

    def __split_url(self, url: str):
        if url.startswith("s3://"):
            url = url.replace("s3://", "")
        bucket, key = url.split("/", maxsplit=1)
        return bucket, key

    def download(self, url: str, local_dir: str):
        bucket, key = self.__split_url(url)
        filename = os.path.basename(key)
        file_path = os.path.join(local_dir, filename)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        meta_data = self.client.head_object(Bucket=bucket, Key=key)
        total_length = int(meta_data.get("ContentLength", 0))

        downloaded = 0

        def progress(chunk):
            nonlocal downloaded
            downloaded += chunk
            done = int(50 * downloaded / total_length)
            sys.stdout.write(
                "\r{}[{}{}]".format(file_path, "█" * done, "." * (50 - done))
            )
            sys.stdout.flush()

        try:
            with open(file_path, "wb") as f:
                self.client.download_fileobj(bucket, key, f, Callback=progress)
            sys.stdout.write("\n")
            sys.stdout.flush()
        except:
            raise Exception(f"downloading file is failed. {url}")
        return file_path


def download(url, chksum=None, cachedir=".cache"):
    cachedir_full = os.path.join(os.getcwd(), cachedir)
    os.makedirs(cachedir_full, exist_ok=True)
    filename = os.path.basename(url)
    file_path = os.path.join(cachedir_full, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path, "rb").read()).hexdigest()[:10] == chksum:
            print(f"using cached model. {file_path}")
            return file_path, True

    s3 = AwsS3Downloader()
    file_path = s3.download(url, cachedir_full)
    if chksum:
        assert (
            chksum == hashlib.md5(open(file_path, "rb").read()).hexdigest()[:10]
        ), "corrupted file!"
    return file_path, False


def get_pytorch_kobart_model(ctx="cpu", cachedir=".cache"):
    pytorch_kobart = {
        "url": "s3://skt-lsl-nlp-model/KoBART/models/kobart_base_cased_ff4bda5738.zip",
        "chksum": "ff4bda5738",
    }
    model_zip, is_cached = download(
        pytorch_kobart["url"], pytorch_kobart["chksum"], cachedir=cachedir
    )
    cachedir_full = os.path.join(os.getcwd(), cachedir)
    model_path = os.path.join(cachedir_full, "kobart_from_pretrained")
    if not os.path.exists(model_path) or not is_cached:
        if not is_cached:
            shutil.rmtree(model_path, ignore_errors=True)
        zipf = ZipFile(os.path.expanduser(model_zip))
        zipf.extractall(path=cachedir_full)
    return model_path


def get_kobart_tokenizer(cachedir=".cache"):
    """Get KoGPT2 Tokenizer file path after downloading"""
    tokenizer = {
        "url": "s3://skt-lsl-nlp-model/KoBART/tokenizers/kobart_base_tokenizer_cased_cf74400bce.zip",
        "chksum": "cf74400bce",
    }
    file_path, is_cached = download(
        tokenizer["url"], tokenizer["chksum"], cachedir=cachedir
    )
    cachedir_full = os.path.expanduser(cachedir)
    if (
        not os.path.exists(os.path.join(cachedir_full, "emji_tokenizer"))
        or not is_cached
    ):
        if not is_cached:
            shutil.rmtree(
                os.path.join(cachedir_full, "emji_tokenizer"), ignore_errors=True
            )
        zipf = ZipFile(os.path.expanduser(file_path))
        zipf.extractall(path=cachedir_full)
    tok_path = os.path.join(cachedir_full, "emji_tokenizer/model.json")
    tokenizer_obj = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    return tokenizer_obj


if __name__ == "__main__":
    from transformers import BartModel, PreTrainedTokenizerFast

    # kobart_tokenizer = get_kobart_tokenizer()
    kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    print(kobart_tokenizer.tokenize("안녕하세요. 한국어 BART 입니다.🤣:)l^o"))

    # model = BartModel.from_pretrained(get_pytorch_kobart_model())
    model = BartModel.from_pretrained('gogamza/kobart-base-v1')
    inputs = kobart_tokenizer(["안녕하세요."], return_tensors="pt")
    print(model(inputs["input_ids"]))
