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


@pytest.mark.parametrize('config_path', glob(os.path.join(CONFIG_DIR, 'huggingface', '*_example.yaml')))
def test_huggingface_caller(config_path: str):
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


def test_alpaca_lora_caller():
    config_path = CONFIG_DIR / 'alpaca_lora' / 'alpaca_lora_llm_example.yaml'
    caller = AlpacaLoraCaller(config=config_path)
    results = caller.generate(PROMPTS)
    print(results)
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()