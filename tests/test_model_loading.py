import os
from glob import glob
from pathlib import Path

import pytest
import torch

from talkative_llm.llm import (AlpacaLoraCaller, CohereCaller,
                               HuggingFaceCaller, MPTCaller, OpenAICaller)

CONFIG_DIR = Path(__file__).parent.parent / 'configs'


@pytest.mark.parametrize('config_path', glob(os.path.join(CONFIG_DIR, 'openai', '*_example.yaml')))
def test_openai_caller(config_path: str):
    print(f'Testing {os.path.basename(config_path)}')
    caller = OpenAICaller(config=config_path, batch_size=30)
    assert caller.config['batch_size'] == 30
    caller.update_caller_params({'temperature': 0.85})
    assert caller.caller_params['temperature'] == 0.85
    del caller


def test_cohere_caller():
    config_path = CONFIG_DIR / 'cohere' / 'cohere_llm_example.yaml'
    caller = CohereCaller(config=config_path, batch_size=30)
    assert caller.config['batch_size'] == 30
    caller.update_caller_params({'temperature': 0.85})
    assert caller.caller_params['temperature'] == 0.85
    del caller


@pytest.mark.parametrize('config_path', glob(os.path.join(CONFIG_DIR, 'huggingface', '*_example.yaml')))
def test_huggingface_caller(config_path: str):
    print(f'Testing {os.path.basename(config_path)}')
    caller = HuggingFaceCaller(config=config_path, batch_size=30)
    assert caller.config['batch_size'] == 30
    caller.update_caller_params({'temperature': 0.85})
    assert caller.caller_params['temperature'] == 0.85
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_mpt_caller():
    config_path = CONFIG_DIR / 'mpt' / 'mpt_llm_example.yaml'
    caller = MPTCaller(config=config_path, batch_size=5)
    assert caller.config['batch_size'] == 5
    caller.update_caller_params({'temperature': 0.85})
    assert caller.caller_params['temperature'] == 0.85
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_alpaca_lora_caller():
    config_path = CONFIG_DIR / 'alpaca_lora' / 'alpaca_lora_llm_example.yaml'
    caller = AlpacaLoraCaller(config=config_path, batch_size=5)
    assert caller.config['batch_size'] == 5
    caller.update_caller_params({'temperature': 0.85})
    assert caller.caller_params['temperature'] == 0.85
    del caller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()