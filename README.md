# talkative-llm

![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)

<div align="center">
  <img src="assets/image/favicon.png" alt="favicon" width="120px">
</div>

## Overview
`talkative-llm` provides a wrapper around various large language models so that inference (i.e., generation) can be done in a coherent manner.
The development of `talkative-llm` is mainly led by [Zae Myung Kim](https://zaemyung.github.io/) and [Jong Inn Park](https://github.com/jong-inn/). Please contact Zae via email (kim01756@umn.edu) or Slack if you have questions. 

## Contents
- [talkative-llm](#talkative-llm)
  - [Overview](#overview)
  - [Contents](#contents)
  - [Model List](#model-list)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Examples](#examples)
    - [OpenAI's Completion (GPT3)](#openais-completion-gpt3)
    - [OpenAI's ChatCompletion (ChatGPT)](#openais-chatcompletion-chatgpt)
  - [Model Weights](#model-weights)
      - [LLaMA 1](#llama-1)
      - [Baize](#baize)
      - [Koala](#koala)
      - [Vicuna](#vicuna)
  - [API Keys](#api-keys)
      - [OpenAI](#openai)
      - [Cohere](#cohere)
  - [Development Process](#development-process)


## Model List

| Name                                                                               | Config Examples | First Release Time | Platform |
|:-----------------------------------------------------------------------------------|:----------------|-------------------:|:---------------|
|[Alpaca-LoRA](https://github.com/tloen/alpaca-lora)                                 | [alpaca_lora_llm_example.yaml](/configs/alpaca_lora/alpaca_lora_llm_example.yaml) | Mar-2023           | AlpacaLoraCaller       |
|[Baize](https://github.com/project-baize/baize-chatbot)                             | [baize_llm_example.yaml](/configs/huggingface/baize_llm_example.yaml) | Apr-2023           | HuggingFaceCaller      |
|[BLOOM](https://huggingface.co/bigscience/bloom)                                    | [bloom_llm_example.yaml](/configs/huggingface/bloom_llm_example.yaml) | May-2022           | HuggingFaceCaller      |
|[ChatGPT](https://platform.openai.com/docs/api-reference/introduction)              | <li>[open_ai_chat_example.yaml](/configs/openai/open_ai_chat_example.yaml)</li> <li>[open_ai_completion_example.yaml](/configs/openai/open_ai_completion_example.yaml)</li> | Nov-2022           | OpenAICaller    |
|[Cohere](https://docs.cohere.com/reference/about)                                   | [cohere_llm_example.yaml](/configs/cohere/cohere_llm_example.yaml) | Jan-2023           | CohereCaller           |
|[Dolly](https://github.com/databrickslabs/dolly)                                    | [dolly_llm_example.yaml](/configs/huggingface/dolly_llm_example.yaml) | Mar-2023           | HuggingFaceCaller      |
|[Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)                          | [falcon_llm_example.yaml](/configs/huggingface/falcon_llm_example.yaml) | May-2023           | HuggingFaceCaller      |
|[Flan-T5](https://huggingface.co/google/flan-t5-base)                               | [huggingface_llm_example.yaml](/configs/huggingface/huggingface_llm_example.yaml) | Oct-2022           | HuggingFaceCaller      |
|[Flan-UL2](https://huggingface.co/google/flan-ul2)                                  | [flan_ul2_llm_example.yaml](/configs/huggingface/flan_ul2_llm_exampleyaml.yaml) | May-2022           | HuggingFaceCaller      |
|[GPT3](https://openai.com/blog/gpt-3-apps)                                          | <li>[open_ai_chat_example.yaml](/configs/huggingface/open_ai_chat_example.yaml)</li> <li>[open_ai_completion_example.yaml](/configs/huggingface/open_ai_completion_example.yaml)</li> | Jun-2020           | OpenAICaller    |
|[GPT4](https://openai.com/gpt-4)                                                    |  | Mar-2023        | OpenAICaller    |
|[GPT4All](https://github.com/nomic-ai/gpt4all)                                      | [gpt4all_llm_example.yaml](/configs/huggingface/gpt4all_llm_example.yaml) | Apr-2023           | HuggingFaceCaller      |
|[gpt4-x-alpaca](https://huggingface.co/chavinlo/gpt4-x-alpaca)                      | [gpt4-x-alpaca_llm_example.yaml](/configs/huggingface/gpt4-x-alpaca_llm_example.yaml) | Mar-2023           | HuggingFaceCaller      |
|[Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)                           | [koala_llm_example.yaml](/configs/huggingface/koala_llm_example.yaml) | Apr-2023           | HuggingFaceCaller      |
|[Lamini-LM](https://github.com/mbzuai-nlp/LaMini-LM)                                | [lamini_llm_example.yaml](/configs/huggingface/lamini_llm_example.yaml) | Apr-2023           | HuggingFaceCaller      |
|[LLaMA](https://github.com/facebookresearch/llama)                                  | [llama_llm_example.yaml](/configs/huggingface/llama_llm_example.yaml) | Feb-2023           | HuggingFaceCaller      |
|[MPT](https://github.com/mosaicml/llm-foundry)                                      | [mpt_llm_example.yaml](/configs/mpt/mpt_llm_example.yaml) | May-2023           | MPTCaller              |
|[Open Alpaca](https://github.com/yxuansu/OpenAlpaca)                                | [openalpaca_llm_example.yaml](/configs/huggingface/openalpaca_llm_example.yaml) | May-2023           | HuggingFaceCaller      |
|[Open Assistant](https://github.com/LAION-AI/Open-Assistant)                        | [open_assist_llm_example.yaml](/configs/huggingface/open_assist_llm_example.yaml) | Apr-2023           | HuggingFaceCaller      |
|[OPT](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)           | [opt_llm_example.yaml](/configs/huggingface/opt_llm_example.yaml) | May-2022           | HuggingFaceCaller      |
|[PandaLM](https://github.com/WeOpenML/PandaLM)                                      | [pandalm_llm_example.yaml](/configs/huggingface/pandalm_llm_example.yaml) | Apr-2023           | HuggingFaceCaller      |
|[PEGASUS](https://github.com/google-research/pegasus)                               | [pegasus_llm_example.yaml](/configs/huggingface/pegasus_llm_example.yaml) | Dec-2019           | HuggingFaceCaller      |
|[Pythia](https://github.com/EleutherAI/pythia)                                      | [pythia_llm_example.yaml](/configs/huggingface/pythia_llm_example.yaml) | Apr-2023           | HuggingFaceCaller      |
|[RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct)   | [redpajama_llm_example.yaml](/configs/huggingface/redpajama_llm_example.yaml) | May-2023           | HuggingFaceCaller      |
|[ReplitLM](https://github.com/replit/ReplitLM)                                      | [replit_llm_example.yaml](/configs/huggingface/replit_llm_example.yaml) | May-2023           | HuggingFaceCaller      |
|[StableLM](https://github.com/Stability-AI/StableLM)                                |  | Apr-2023           | HuggingFaceCaller      |
|[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                     | [stanford_alpaca_llm_example.yaml](/configs/huggingface/stanford_alpaca_llm_example.yaml) | Mar-2023           | HuggingFaceCaller      |
|[T5](https://github.com/google-research/text-to-text-transfer-transformer)          | [t5_llm_example.yaml](/configs/huggingface/t5_llm_example.yaml) | Apr-2020           | HuggingFaceCaller      |
|[UL2](https://huggingface.co/google/ul2)                                            | [ul2_llm_example.yaml](/configs/huggingface/ul2_llm_example.yaml) | May-2022           | HuggingFaceCaller      |
|[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)                                 | <li>[vicuna_llm_7b_example.yaml](/configs/huggingface/vicuna_llm_7b_example.yaml)</li> <li>[vicuna_llm_13b_example.yaml](/configs/huggingface/vicuna_llm_13b_example.yaml)</li> | Mar-2023           | HuggingFaceCaller      |


## Setup

Clone the repo and make a venv.

```
git clone https://github.com/minnesotanlp/talkative-llm/
python -m venv talkative-llm-venv
source talkative-llm-venv/bin/activate
cd talkative-llm
```

Install the repo.

```
pip install -e .
```

If you want to use cutting-edge features, use develop branch.

```
git checkout develop
```

You are all set!


## Usage

```python
import os
from glob import glob
from pathlib import Path

import pytest
import torch

from talkative_llm.llm import (AlpacaLoraCaller, CohereCaller,
                               HuggingFaceCaller, MPTCaller, OpenAICaller)

CONFIG_DIR = Path(__file__).parent.parent / 'configs'

PROMPTS = ['Who won the world series in 2020?',
           'Knock, Knock. Who\'s there?',
           'What makes a great dish?']


def test_openai_caller_completion():
    config_path = CONFIG_DIR / 'openai' / 'openai_completion_example.yaml'
    caller = OpenAICaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller


def test_openai_caller_chat():
    config_path = CONFIG_DIR / 'openai' / 'openai_chat_example.yaml'
    caller = OpenAICaller(config=config_path)
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who won the world series in 2020?'},
        {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020.'},
        {'role': 'user', 'content': 'Where was it played?'}
    ]
    results = caller.generate(inputs=messages)
    print(results)
    del caller


def test_cohere_caller():
    config_path = CONFIG_DIR / 'cohere' / 'cohere_llm_example.yaml'
    caller = CohereCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller


def test_huggingface_caller(config_path: str):
    print(f'Testing {os.path.basename(config_path)}')
    caller = HuggingFaceCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_huggingface_caller():
    config_path = os.path.join(CONFIG_DIR, 'huggingface', 'huggingface_llm_example.yaml')
    print(f'Testing {os.path.basename(config_path)}')
    caller = HuggingFaceCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_mpt_caller():
    config_path = CONFIG_DIR / 'mpt' / 'mpt_llm_example.yaml'
    caller = MPTCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```


## Model Weights

First, you need to obtain the LLaMA 1 weight from [facebookresearch](https://github.com/facebookresearch/llama/tree/llama_v1).
You can get LLaMA 2 weight from [here](https://github.com/facebookresearch/llama).

The models below need weight conversions from LLaMA 1 weight. After the conversions, they need to be located locally in your environment.
#### [LLaMA 1](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)

#### [Baize](https://github.com/project-baize/baize-chatbot)

#### [Koala](https://github.com/young-geng/EasyLM/blob/main/docs/koala.md)

#### [Vicuna](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md)

## API Keys

OpenAI and Cohere need API Keys from their web page. Please get them from links below.

#### [OpenAI](https://platform.openai.com/docs/introduction/overview)

#### [Cohere](https://docs.cohere.com/docs?utm_term=&utm_campaign=Cohere+Brand+%26+Industry+Terms&utm_source=adwords&utm_medium=ppc&hsa_acc=4946693046&hsa_cam=20368816223&hsa_grp=154209120409&hsa_ad=666081801359&hsa_src=g&hsa_tgt=dsa-19959388920&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gad=1&gclid=Cj0KCQjwoK2mBhDzARIsADGbjeqZtpfA-FMKldT_EsCl0bE7z8ZSri4Cu-Jp2wsumHVNUVLW-3XALBoaAtTvEALw_wcB)

Before using, create `.env` for saving all your keys. This file will be ignored by git.
  - Keys to save (Just keep an empty string for unused ones):
    - `COHERE_API_KEY`
    - `OPENAI_API_KEY`
    - `OPENAI_ORGANIZATION_ID`

## Development Process
- Push to `develop` branch directly if your changes don't require code reviews. Otherwise, create a new branch and do pull-requests to the `develop` branch.
- Once enough features are implemented to `develop` branch, it will be peer-reviewd and merged to `main` branch.
 
## Contributors

* [Zae Myung Kim](https://zaemyung.github.io/)
* [Jong Inn Park](https://github.com/jong-inn/)





