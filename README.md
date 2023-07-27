# talkative-llm

![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)

- A library to query large language models (LLMs) given a file of prompts.
- Supported LLMs can be found [here](https://docs.google.com/spreadsheets/d/1CSA52gXEOIkmzwj78jT50zQCzIS3uV5gkK66akvBHkk/edit#gid=0).
- Clone, (make venv), and do `pip install -e .` for editable installation.
  - You may want to `git checkout` and use `develop` branch for cutting-edge features.
- LLMs such as LLaMA, Baize, and Vicuna require pre-downloaded weights which can be found at `cs-u-eagle.cs.umn.edu:/home/jonginn/volume/models`
  - Bigger LLaMA weights are `cs-u-elk.cs.umn.edu:/home/zaemyung/Models/LLaMA`
- Before using, create `.env` for saving all your keys. This file will be ignored by git.
  - Keys to save (Just keep an empty string for unused ones):
    - `COHERE_API_KEY`
    - `OPENAI_API_KEY`
    - `OPENAI_ORGANIZATION_ID`

### Development process
- Push to `develop` branch directly if your changes don't require code reviews. Otherwise, create a new branch and do pull-requests to the `develop` branch.
- Once enough features are implemented to `develop` branch, it will be peer-reviewd and merged to `main` branch.

## Model List

| Name                                                                               | Implemented   | First Release Time | Current Version |
|:-----------------------------------------------------------------------------------|:-------------:|-------------------:|:---------------:|
|[Alpaca-LoRA](https://github.com/tloen/alpaca-lora)                                 | <li>[x] </li> | Mar-2023           | -               |
|[Baize](https://github.com/project-baize/baize-chatbot)                             | <li>[x] </li> | Apr-2023           | v2              |
|Bard                                                                                | **TBD**       |                    |                 |
|[BLOOM](https://huggingface.co/bigscience/bloom)                                    | <li>[x] </li> | May-2022           | v1.3            |
|[ChatGPT](https://platform.openai.com/docs/api-reference/introduction)              | <li>[x] </li> | Nov-2022           | -               |
|[Cohere](https://docs.cohere.com/reference/about)                                   | <li>[x] </li> | Jan-2023           | command-nightly |
|[Colossal Chat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)| <li>[ ] </li> | Mar-2023           | v1.0.0          |
|[Dolly](https://github.com/databrickslabs/dolly)                                    | <li>[x] </li> | Mar-2023           | v2              |
|[Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)                          | <li>[x] </li> | May-2023           | -               |
|[Flan-T5](https://huggingface.co/google/flan-t5-base)                               | <li>[x] </li> | Oct-2022           | -               |
|[Flan-UL2](https://huggingface.co/google/flan-ul2)                                  | <li>[x] </li> | May-2022           | -               |
|[GPT3](https://openai.com/blog/gpt-3-apps)                                          | <li>[x] </li> | Jun-2020           | 3.5-turbo       |
|[GPT4](https://openai.com/gpt-4)                                                    | <li>[ ] </li> | Mar-2023           | 32k-0613        | 
|[GPT4All](https://github.com/nomic-ai/gpt4all)                                      | <li>[x] </li> | Apr-2023           | -               |
|[gpt4-x-alpaca](https://huggingface.co/chavinlo/gpt4-x-alpaca)                      | <li>[x] </li> | Mar-2023           | -               |
|[Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)                           | <li>[x] </li> | Apr-2023           | -               |
|[Lamini-LM](https://github.com/mbzuai-nlp/LaMini-LM)                                | <li>[x] </li> | Apr-2023           | -               |
|[LLaMA](https://github.com/facebookresearch/llama)                                  | <li>[x] </li> | Feb-2023           | v2              |
|[MPT](https://github.com/mosaicml/llm-foundry)                                      | <li>[x] </li> | May-2023           | v0.2.0          |
|[Open Alpaca](https://github.com/yxuansu/OpenAlpaca)                                | <li>[x] </li> | May-2023           | -               |
|[Open Assistant](https://github.com/LAION-AI/Open-Assistant)                        | <li>[x] </li> | Apr-2023           | v0.0.3-alpha35  |
|[OPT](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)           | <li>[x] </li> | May-2022           | -               |
|[PandaLM](https://github.com/WeOpenML/PandaLM)                                      | <li>[x] </li> | Apr-2023           | -               |
|[PEGASUS](https://github.com/google-research/pegasus)                               | <li>[x] </li> | Dec-2019           | -               |
|[Pythia](https://github.com/EleutherAI/pythia)                                      | <li>[x] </li> | Apr-2023           | v0              |
|[RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct)   | <li>[x] </li> | May-2023           | -               |
|[ReplitLM](https://github.com/replit/ReplitLM)                                      | <li>[x] </li> | May-2023           | v1              |
|[StableLM](https://github.com/Stability-AI/StableLM)                                | <li>[x] </li> | Apr-2023           | -               |
|[StackLLaMA](https://huggingface.co/blog/stackllama)                                | **TBD**       |                    |                 |
|[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)                     | <li>[x] </li> | Mar-2023           | -               |
|[T5](https://github.com/google-research/text-to-text-transfer-transformer)          | <li>[x] </li> | Apr-2020           | v0.4.0          |
|[UL2](https://huggingface.co/google/ul2)                                            | <li>[x] </li> | May-2022           | -               |
|[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)                                 | <li>[x] </li> | Mar-2023           | v1.3            |


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

### OpenAI's Completion (GPT3)
<img width="1276" alt="image" src="https://user-images.githubusercontent.com/3746478/226238549-14f01831-d709-4657-bf9a-749774a31730.png">

### OpenAI's ChatCompletion (ChatGPT)
<img width="1279" alt="image" src="https://user-images.githubusercontent.com/3746478/226238673-840d4cfa-b26b-449b-abd3-659e2e3365d9.png">
