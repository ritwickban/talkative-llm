import os
from glob import glob
from pathlib import Path

import torch

from talkative_llm.llm import (AlpacaLoraCaller, CohereCaller,
                               HuggingFaceCaller, MPTCaller, OpenAICaller)

CONFIG_DIR = Path() / 'configs'
print(CONFIG_DIR)

PROMPTS = ['Who won the world series in 2020?',
           'Knock, Knock. Who\'s there?',
           'What makes a great dish?']


def openai_caller_completion():
    config_path = CONFIG_DIR / 'openai' / 'openai_completion_example.yaml'
    caller = OpenAICaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller


def openai_caller_chat():
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


def cohere_caller():
    config_path = CONFIG_DIR / 'cohere' / 'cohere_llm_example.yaml'
    api_key= 'GAuYh4XB9FREC58Po0jIY7mcgnKUi0zZP4juQz75'
    caller = CohereCaller(config=config_path, api_key=api_key)
    results = caller.generate(PROMPTS)
    print(results)
    del caller


def huggingface_caller(config_path: str):
    print(f'Testing {os.path.basename(config_path)}')
    caller = HuggingFaceCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def huggingface_caller():
    config_path = os.path.join(CONFIG_DIR, 'huggingface', 'huggingface_llm_example.yaml')
    print(f'Testing {os.path.basename(config_path)}')
    caller = HuggingFaceCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mpt_caller():
    config_path = CONFIG_DIR / 'mpt' / 'mpt_llm_example.yaml'
    caller = MPTCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

cohere_caller()