import importlib
import inspect
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import yaml
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from dotenv import load_dotenv
from huggingface_hub import try_to_load_from_cache
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_random_exponential

console = Console()
error_console = Console(stderr=True, style='bold red')
TENACITY_RETRY_N = 5
load_dotenv()


class LLMCaller(ABC):
    @abstractmethod
    def __init__(self, config: str | Path | Dict[str, Any], **kwargs) -> None:
        console.log(f'{self.__class__.__name__} is instantiated.')
        if (isinstance(config, str) or isinstance(config, Path)) and os.path.isfile(config):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            error_console.log()

        # Update config
        for key, value in kwargs.items():
            if key in self.config:
                previous_value = self.config[key]
                console.log(f'config["{key}"] is updated from {previous_value} to {value}.')
            else:
                console.log(f'config["{key}"] is added with {value}.')
            self.config[key] = value

    @abstractmethod
    def generate(self, inputs: List[str] | List[Dict]) -> List[Dict] | Dict:
        '''This method passes inputs to either an LLM directly or an API and retrieves generated results.

        Args:
            inputs: either (1) a list of prompt strings or,
                    (2) a list of of dict messages in `chat format` as specified by OpenAI.
                    (1) corresponds to generating multiple and separate results for the prompts strings,
                    while (2) refers to generating a single chat response given the previous chat history.

        Returns: (1) a list of dict containing corresponding generated results or,
                 (2) a single dict result in the case of `chat` mode.
        '''
        pass

    def update_caller_params(self, new_caller_params: Dict[str, Any]) -> None:
        for key, value in new_caller_params.items():
            if key in self.caller_params:
                previous_value = self.caller_params[key]
                self.caller_params[key] = value
                console.log(f'caller_params["{key}"] is updated from {previous_value} to {value}.')

    def import_dependencies(self) -> bool:
        loaded_all = True
        for package in self.dependencies:
            try:
                globals()[package] = importlib.import_module(package)
            except ModuleNotFoundError:
                error_console.log(f'Could not load the package: {package}')
                loaded_all = False
        return loaded_all

    def check_dependencies(self) -> None:
        if not self.import_dependencies():
            error_console.log('Could not load all the necessary dependencies. Please install them accordingly.')
            sys.exit()


class OpenAICaller(LLMCaller):
    def __init__(self, config: str | Path | Dict[str, Any], **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        assert self.config['framework'] == 'openai'
        self.dependencies = ['openai']
        self.check_dependencies()

        openai.organization = self.config.get('organization_key', os.environ.get('OPENAI_ORGANIZATION_ID'))
        openai.api_key = self.config.get('api_key', os.environ.get('OPENAI_API_KEY'))

        if openai.organization is None or openai.api_key is None:
            error_console.log(('API keys are not found. '
                               'Please either pass the keys as '
                               '(1) kwargs to OpenAICaller by setting `organization_key` and `api_key`;\n'
                               'or, (2) by setting `OPENAI_ORGANIZATION_ID` and `OPENAI_API_KEY` in an .env file.'
                              ))
            sys.exit()

        mode_to_api_caller = {
            'chat': openai.ChatCompletion,
            'completion': openai.Completion,
        }

        self.mode = self.config['mode']
        if self.mode not in mode_to_api_caller:
            error_console.log(f'Unsupported mode: {self.mode}\nThe supported modes are: {list(mode_to_api_caller.keys())}')
            sys.exit()
        self.caller = mode_to_api_caller[self.mode]
        self.caller_params = self.config['params']
        console.log(f'API parameters are:\n{self.caller_params}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(TENACITY_RETRY_N))
    def generate(self, inputs: List[str] | List[Dict]) -> List[Dict] | Dict:
        if self.mode == 'chat':
            assert isinstance(inputs, list) and isinstance(inputs[0], dict)
            assert 'role' in inputs[0] and 'content' in inputs[0]
            response = self.caller.create(messages=inputs, **self.caller_params)
            generation = response['choices'][0]['message']['content']
            finish_reason = response['choices'][0]['finish_reason']
            result = {'generation': generation, 'finish_reason': finish_reason}
            return result
        elif self.mode == 'completion':
            assert isinstance(inputs[0], str)
            all_results = []
            response = self.caller.create(prompt=inputs, **self.caller_params)
            for choice in response.choices:
                result = {'generation': choice.text, 'finish_reason': choice.finish_reason}
                all_results.append(result)
            return all_results
        else:
            raise NotImplementedError(f'Mode {self.mode} is not implemented yet.')


class CohereCaller(LLMCaller):
    def __init__(self, config: str | Path | Dict[str, Any], **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        assert self.config['framework'] == 'cohere'
        self.dependencies = ['cohere']
        self.check_dependencies()

        api_key = self.config.get('api_key', os.environ.get('COHERE_API_KEY'))
        self.caller = cohere.Client(api_key)
        self.caller_params = self.config['params']
        console.log(f'API parameters are:\n{self.caller_params}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(TENACITY_RETRY_N))
    def generate(self, inputs: List[str]) -> List[Dict]:
        assert isinstance(inputs, list) and isinstance(inputs[0], str)
        responses = self.caller.batch_generate(prompts=inputs, **self.caller_params)
        all_results = []
        for response in responses:
            for generation in response.generations:
                all_results.append({'generation': generation.text})
        return all_results


class HuggingFaceCaller(LLMCaller):
    def __init__(self, config: str | Path | Dict[str, Any], **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        assert self.config['framework'] == 'huggingface'
        self.dependencies = ['torch', 'transformers']
        self.check_dependencies()

        self.skip_special_tokens = self.config['skip_special_tokens']
        self.caller_params = self.config['params']
        if 'cuda' in self.config['device']:
            self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config['device'])
        console.log(f'Current device: {self.device}')

        model_type = getattr(transformers, self.config['mode'])
        model_name = self.config['model'].pop('name')
        for k, v in self.config['model'].items():
            if v == 'torch.bfloat16':
                self.config['model'][k] = torch.bfloat16
        model_params = self.config['model']
        tokenizer_params = self.config.get('tokenizer', {})

        try:
            self.generation_config, unused_kwargs = transformers.GenerationConfig.from_pretrained(model_name,
                                                                                                  **self.caller_params,
                                                                                                  return_unused_kwargs=True)
            if len(unused_kwargs) > 0:
                console.log(f'Following config parameters are ignored in generation, please check:\n{unused_kwargs}')
        except OSError:
            error_console.log(f'`generation_config.json` could not be found at https://huggingface.co/{model_name}')
            # TODO: Need to check if just passing self.caller_params are ok for the generate method.
            self.generation_config = transformers.GenerationConfig(**self.caller_params)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **tokenizer_params)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if 'device_map' in model_params:
            self.model = model_type.from_pretrained(model_name, **model_params)
        else:
            self.model = model_type.from_pretrained(model_name, **model_params).to(self.device)
        self.model.eval()

        console.log(f'Loaded parameters are:\n{self.generation_config}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(TENACITY_RETRY_N))
    def generate(self, inputs: List[str] | List[Dict]) -> List[Dict]:
        tokenized_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True).to(self.device)
        generation_args = set(inspect.signature(self.model.forward).parameters)
        # Remove unused args
        unused_args = [key for key in tokenized_inputs.keys() if key not in generation_args]
        for key in unused_args:
            del tokenized_inputs[key]

        with torch.no_grad():
            outputs = self.model.generate(**tokenized_inputs, generation_config=self.generation_config)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=self.skip_special_tokens)
        all_results = []
        for decoded_output in decoded_outputs:
            all_results.append({'generation': decoded_output})
        return all_results


class MPTCaller(LLMCaller):
    def __init__(self, config: str | Path | Dict[str, Any], **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        assert self.config['framework'] == 'mpt'
        # Need pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
        self.dependencies = ['accelerate', 'einops', 'flash_attn', 'torch', 'transformers']
        self.check_dependencies()

        self.skip_special_tokens = self.config['skip_special_tokens']
        self.caller_params = self.config['params']
        if 'cuda' in self.config['device'] and torch.cuda.is_available():
            self.device = torch.device(self.config['device'])
        else:
            error_console.log('MPTCaller only works with cuda backend.')
            sys.exit()

        console.log(f'Current device: {self.device}')

        model_type = getattr(transformers, self.config['mode'])
        model_name = self.config['model'].pop('name')
        device_map = self.config['model'].pop('device_map')
        attn_impl = self.config['model'].pop('attn_impl')
        max_memory_mapping = self.config['model'].pop('max_memory_mapping')
        self.model_config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # triton is not supported for every Nvidia GPU, in case of error, use 'cuda'
        self.model_config.attn_config['attn_impl'] = attn_impl
        self.model_config.init_device = 'cuda:0' if 'cuda' in self.device.type else 'cpu'
        self.model_config.max_seq_len = 4096

        try:
            self.generation_config, unused_kwargs = transformers.GenerationConfig.from_pretrained(model_name,
                                                                                                  **self.caller_params,
                                                                                                  return_unused_kwargs=True)
            if len(unused_kwargs) > 0:
                console.log(f'Following config parameters are ignored in generation, please check:\n{unused_kwargs}')
        except OSError:
            error_console.log(f'`generation_config.json` could not be found at https://huggingface.co/{model_name}')
            # TODO: Need to check if just passing self.caller_params are ok for the generate method.
            self.generation_config = transformers.GenerationConfig(**self.caller_params)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        self.generation_config.pad_token = self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

        with init_empty_weights():
            model = model_type.from_pretrained(
                model_name,
                config=self.model_config,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        model.tie_weights()
        model_dir = try_to_load_from_cache(repo_id=model_name, filename='config.json')
        weights_location = os.path.join(os.path.dirname(model_dir), 'pytorch_model.bin.index.json')
        self.model = load_checkpoint_and_dispatch(model, weights_location, device_map=device_map, max_memory=max_memory_mapping,
                                                  no_split_module_classes=['MPTBlock'], dtype=torch.float16)
        self.model.eval()

        console.log(f'Loaded parameters are:\n{self.generation_config}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(TENACITY_RETRY_N))
    def generate(self, inputs: List[str] | List[Dict]) -> List[Dict]:
        tokenized_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True).to(self.device)
        generation_args = set(inspect.signature(self.model.forward).parameters)
        # Remove unused args
        unused_args = [key for key in tokenized_inputs.keys() if key not in generation_args]
        for key in unused_args:
            del tokenized_inputs[key]

        with torch.no_grad():
            outputs = self.model.generate(**tokenized_inputs, generation_config=self.generation_config)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=self.skip_special_tokens)
        all_results = []
        for decoded_output in decoded_outputs:
            all_results.append({'generation': decoded_output})
        return all_results


class AlpacaLoraCaller(LLMCaller):
    def __init__(self, config: str | Path | Dict[str, Any], **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        assert self.config['framework'] == 'alpaca-lora'
        self.dependencies = ['peft', 'torch', 'transformers']
        self.check_dependencies()

        self.skip_special_tokens = self.config['skip_special_tokens']
        self.caller_params = self.config['params']
        if 'cuda' in self.config['device'] and torch.cuda.is_available():
            self.device = torch.device(self.config['device'])
        else:
            self.device = torch.device('cpu')

        console.log(f'Current device: {self.device}')

        model_type = getattr(transformers, self.config['mode'])
        model_name = self.config['model'].pop('name')
        lora_weights = self.config['model'].pop('lora_weights')
        load_8bit = self.config['model'].pop('load_8bit')

        try:
            self.generation_config, unused_kwargs = transformers.GenerationConfig.from_pretrained(model_name,
                                                                                                  **self.caller_params,
                                                                                                  return_unused_kwargs=True)
            if len(unused_kwargs) > 0:
                console.log(f'Following config parameters are ignored in generation, please check:\n{unused_kwargs}')
        except OSError:
            error_console.log(f'`generation_config.json` could not be found at https://huggingface.co/{model_name}')
            # TODO: Need to check if just passing self.caller_params are ok for the generate method.
            self.generation_config = transformers.GenerationConfig(**self.caller_params)

        if self.device.type == 'cuda':
            model = model_type.from_pretrained(model_name, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map='auto')
            self.model = peft.PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16, device_map='auto')
        else:
            model = model_type.from_pretrained(model_name, device_map='auto', low_cpu_mem_usage=True)
            self.model = peft.PeftModel.from_pretrained(model, lora_weights, device_map='auto')

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if not load_8bit:
            self.model.half()

        self.model.eval()

        console.log(f'Loaded parameters are:\n{self.generation_config}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(TENACITY_RETRY_N))
    def generate(self, inputs: List[str] | List[Dict]) -> List[Dict]:
        tokenized_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True).to(self.device)
        generation_args = set(inspect.signature(self.model.forward).parameters)
        # Remove unused args
        unused_args = [key for key in tokenized_inputs.keys() if key not in generation_args]
        for key in unused_args:
            del tokenized_inputs[key]

        with torch.no_grad():
            outputs = self.model.generate(**tokenized_inputs, generation_config=self.generation_config)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=self.skip_special_tokens)
        all_results = []
        for decoded_output in decoded_outputs:
            all_results.append({'generation': decoded_output})
        return all_results


def get_supported_llm(config: Dict) -> LLMCaller:
    framework = config['framework']
    if  framework == 'openai':
        return OpenAICaller(config)
    elif framework == 'huggingface':
        return HuggingFaceCaller(config)
    elif framework == 'cohere':
        return CohereCaller(config)
    elif framework == 'alpaca-lora':
        return AlpacaLoraCaller(config)
    elif framework == 'mpt':
        return MPTCaller(config)
    else:
        error_console.log(f'Unsupported framework: {framework}')
        sys.exit(1)