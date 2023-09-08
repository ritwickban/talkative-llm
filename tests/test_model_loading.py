from pathlib import Path

from talkative_llm.llm import (AlpacaLoraCaller, CohereCaller,
                               HuggingFaceCaller, MPTCaller, OpenAICaller)

CONFIG_DIR = Path(__file__).parent.parent / 'configs'


def test_openai_caller():
    config_names = ['openai_completion_example.yaml', 'openai_chat_example.yaml']
    for config_name in config_names:
        config_path = CONFIG_DIR / 'openai' / config_name
        caller = OpenAICaller(config=config_path, batch_size=30)
        assert caller.config['batch_size'] == 30
        caller.update_caller_params({'temperature': 0.85})
        assert caller.caller_params['temperature'] == 0.85


def test_cohere_caller():
    config_path = CONFIG_DIR / 'cohere' / 'cohere_llm_example.yaml'
    caller = CohereCaller(config=config_path, batch_size=30)
    assert caller.config['batch_size'] == 30
    caller.update_caller_params({'temperature': 0.85})
    assert caller.caller_params['temperature'] == 0.85


# def test_huggingface_caller():
#     config_path = CONFIG_DIR / 'huggingface' / 'huggingface_llm_example.yaml'
#     caller = HuggingFaceCaller(config=config_path, batch_size=30)
#     assert caller.config['batch_size'] == 30
#     caller.update_caller_params({'temperature': 0.85})
#     assert caller.caller_params['temperature'] == 0.85


# def test_mpt_caller():
#     config_path = CONFIG_DIR / 'mpt' / 'mpt_llm_example.yaml'
#     caller = MPTCaller(config=config_path, batch_size=5)
#     assert caller.config['batch_size'] == 5
#     caller.update_caller_params({'temperature': 0.85})
#     assert caller.caller_params['temperature'] == 0.85


# def test_alpaca_lora_caller():
#     config_path = CONFIG_DIR / 'alpaca_lora' / 'alpaca_lora_llm_example.yaml'
#     caller = AlpacaLoraCaller(config=config_path, batch_size=5)
#     assert caller.config['batch_size'] == 5
#     caller.update_caller_params({'temperature': 0.85})
#     assert caller.caller_params['temperature'] == 0.85