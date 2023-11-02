# talkative-llm

![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)

<div align="center">
  <img src="assets/image/favicon.png" alt="favicon" width="120px">
</div>

## Overview
Introducing `talkative-llm`, our new platform that makes Large Language Model inferences accessible to everyone!

With `talkative-llm`, you can use the latest and greatest LLMs with just a few clicks. Our easy-to-use interface and simple config file make it possible for anyone to get started with LLMs, regardless of their experience level.


## Contents
- [Model List](#model-list)
- [Setup](#setup)
- [Usage](#usage)
- [Examples](#examples)
- [Model Weights](#model-weights)
- [API Keys](#api-keys)
- [Development Process](#development-process)


## Model List

| Name                                                                               | Config Examples | First Release Time | Platform |
|:-----------------------------------------------------------------------------------|:----------------|-------------------:|:---------------|
|[Alpaca-LoRA](https://github.com/tloen/alpaca-lora)                                 | [alpaca_lora_example.yml](/configs/alpaca_lora_example.yml) | Mar-2023           | AlpacaLoraCaller       |
|[Baize](https://github.com/project-baize/baize-chatbot)                             | [baize_llm_example.yml](/configs/baize_llm_example.yml) | Apr-2023           | HuggingFaceCaller      |
|[BLOOM](https://huggingface.co/bigscience/bloom)                                    | [bloom_llm_example.yml](/configs/bloom_llm_example.yml) | May-2022           | HuggingFaceCaller      |
|[ChatGPT](https://platform.openai.com/docs/api-reference/introduction)              | <li>[open_ai_chat_example.yaml](/configs/open_ai_chat_example.yaml)</li> <li>[open_ai_completion_example.yaml](/configs/open_ai_completion_example.yaml)</li> | Nov-2022           | OpenAICaller    |
|[Cohere](https://docs.cohere.com/reference/about)                                   | [cohere_llm_example.yaml](/configs/cohere_llm_example.yaml) | Jan-2023           | CohereCaller           |
|[Dolly](https://github.com/databrickslabs/dolly)                                    | [dolly_llm_example.yml](/configs/dolly_llm_example.yml) | Mar-2023           | HuggingFaceCaller      |
|[Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)                          | [falcon_llm_example.yml](/configs/falcon_llm_example.yml) | May-2023           | HuggingFaceCaller      |
|[Flan-T5](https://huggingface.co/google/flan-t5-base)                               | [huggingface_llm_example.yaml](/configs/huggingface_llm_example.yaml) | Oct-2022           | HuggingFaceCaller      |
|[Flan-UL2](https://huggingface.co/google/flan-ul2)                                  | [flan_ul2_llm_exampleyml.yml](/configs/flan_ul2_llm_exampleyml.yml) | May-2022           | HuggingFaceCaller      |
|[GPT3](https://openai.com/blog/gpt-3-apps)                                          | <li>[open_ai_chat_example.yaml](/configs/open_ai_chat_example.yaml)</li> <li>[open_ai_completion_example.yaml](/configs/open_ai_completion_example.yaml)</li> | Jun-2020           | OpenAICaller    |
|[GPT4](https://openai.com/gpt-4)                                                    |  | Mar-2023        | OpenAICaller    |
|[GPT4All](https://github.com/nomic-ai/gpt4all)                                      | [gpt4all_llm_example.yml](/configs/gpt4all_llm_example.yml) | Apr-2023           | HuggingFaceCaller      |
|[gpt4-x-alpaca](https://huggingface.co/chavinlo/gpt4-x-alpaca)                      | [gpt4-x-alpaca_llm_example.yml](/configs/gpt4-x-alpaca_llm_example.yml) | Mar-2023           | HuggingFaceCaller      |
|[Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)                           | [koala_llm_example.yml](/configs/koala_llm_example.yml) | Apr-2023           | HuggingFaceCaller      |
|[Lamini-LM](https://github.com/mbzuai-nlp/LaMini-LM)                                | [lamini_llm_example.yml](/configs/lamini_llm_example.yml) | Apr-2023           | HuggingFaceCaller      |
|[LLaMA](https://github.com/facebookresearch/llama)                                  | [llama_llm_example.yml](/configs/llama_llm_example.yml) | Feb-2023           | HuggingFaceCaller      |
|[MPT](https://github.com/mosaicml/llm-foundry)                                      | [mpt_llm_example.yml](/configs/mpt_llm_example.yml) | May-2023           | MPTCaller              |
|[Open Alpaca](https://github.com/yxuansu/OpenAlpaca)                                | [openalpaca_llm_example.yml](/configs/openalpaca_llm_example.yml) | May-2023           | HuggingFaceCaller      |
|[Open Assistant](https://github.com/LAION-AI/Open-Assistant)                        | [open_assist_llm_example.yml](/configs/open_assist_llm_example.yml) | Apr-2023           | HuggingFaceCaller      |
|[OPT](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)           | [opt_llm_example.yml](/configs/opt_llm_example.yml) | May-2022           | HuggingFaceCaller      |
|[PandaLM](https://github.com/WeOpenML/PandaLM)                                      | [pandalm_llm_example.yml](/configs/pandalm_llm_example.yml) | Apr-2023           | HuggingFaceCaller      |
|[PEGASUS](https://github.com/google-research/pegasus)                               | [pegasus_llm_example.yml](/configs/pegasus_llm_example.yml) | Dec-2019           | HuggingFaceCaller      |
|[Pythia](https://github.com/EleutherAI/pythia)                                      | [pythia_llm_example.yml](/configs/pythia_llm_example.yml) | Apr-2023           | HuggingFaceCaller      |
|[RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct)   | [redpajama_llm_example.yml](/configs/redpajama_llm_example.yml) | May-2023           | HuggingFaceCaller      |
|[ReplitLM](https://github.com/replit/ReplitLM)                                      | [replit_llm_example.yml](/configs/replit_llm_example.yml) | May-2023           | HuggingFaceCaller      |
|[StableLM](https://github.com/Stability-AI/StableLM)                                |  | Apr-2023           | HuggingFaceCaller      |
|[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                     | [stanford_alpaca_llm_example.yml](/configs/stanford_alpaca_llm_example.yml) | Mar-2023           | HuggingFaceCaller      |
|[T5](https://github.com/google-research/text-to-text-transfer-transformer)          | [t5_llm_example.yml](/configs/t5_llm_example.yml) | Apr-2020           | HuggingFaceCaller      |
|[UL2](https://huggingface.co/google/ul2)                                            | [ul2_llm_example.yml](/configs/ul2_llm_example.yml) | May-2022           | HuggingFaceCaller      |
|[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)                                 | <li>[vicuna_llm_7b_example.yml](/configs/vicuna_llm_7b_example.yml)</li> <li>[vicuna_llm_13b_example.yml](/configs/vicuna_llm_13b_example.yml)</li> | Mar-2023           | HuggingFaceCaller      |


## Setup

Clone the repo and make a venv.

```
git clone https://github.com/minnesotanlp/talkative-llm/
python -m venv talkative-llm-venv
source talkative-llm-venv/bin/activate
cd talkative-llm
```

Install requirements.

```
pip install -r requirements.txt
```

If you want to use cutting-edge features, use develop branch.

```
git checkout develop
```

You are all set!


## Usage

```
Usage: talkative_llm [-h] [-v] -c CONFIG -p PROMPT [-o OUTPUT] [--delay-in-seconds DELAY_IN_SECONDS]

Python library for querying large language models

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -c CONFIG, --config CONFIG
                        config file for language model to be called
  -p PROMPT, --prompt PROMPT
                        either path to an input JSONL prompt file or a single prompt string
  -o OUTPUT, --output OUTPUT
                        path to output file, if not set, print to stdout
  --delay-in-seconds DELAY_IN_SECONDS
                        delay in seconds for each OpenAI API call
```

An example line of an input JSONL file for `chat` mode - each line is a JSON:
```json
[{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "My name is Harry Potter."}, {"role": "assistant", "content": "Hello, there."}, {"role": "user", "content": "What is my name?"}]
```

An example line of an input JSONL file for any other mode - each line is a JSON:
```json
{"prompt": "Tell me a joke."}
```

Run as:
```bash
$ talkative_llm -c openai_gpt-3.5-turbo.yaml -p test_prompt.json -o ./out.json
```

## Examples

### OpenAI's Completion (GPT3)
<img width="1276" alt="image" src="https://user-images.githubusercontent.com/3746478/226238549-14f01831-d709-4657-bf9a-749774a31730.png">

### OpenAI's ChatCompletion (ChatGPT)
<img width="1279" alt="image" src="https://user-images.githubusercontent.com/3746478/226238673-840d4cfa-b26b-449b-abd3-659e2e3365d9.png">


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