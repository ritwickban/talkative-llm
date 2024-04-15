import os
from glob import glob
from pathlib import Path

import torch

from talkative_llm.llm import (AlpacaLoraCaller, CohereCaller,
                               HuggingFaceCaller, MPTCaller, OpenAICaller)

CONFIG_DIR = Path(__file__).parent / 'configs'

PROMPTS = ['Has any U.S. president ever served more than 2 terms?',
           'How do plants create energy from the sun?',
           'What is H20?',
           'What is Newton\'s first law?',
           'Where did life begin on earth?']


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
    caller = CohereCaller(config=config_path)
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


# def huggingface_caller():
#     config_path = os.path.join(CONFIG_DIR, 'huggingface', 'huggingface_llm_example.yaml')
#     print(config_path)
#     print(f'Testing {os.path.basename(config_path)}')
#     caller = HuggingFaceCaller(config=config_path)
#     results = caller.generate(PROMPTS)
#     print(results)
#     del caller
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


def mpt_caller():
    config_path = CONFIG_DIR / 'mpt' / 'mpt_llm_example.yaml'
    caller = MPTCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


config_path = os.path.join(CONFIG_DIR, 'huggingface', 'huggingface_llm_example.yaml')
huggingface_caller(config_path)

# config_path = os.path.join(CONFIG_DIR, 'huggingface', 'lamini_llm_example.yaml')
# huggingface_caller(config_path)

cohere_caller()
