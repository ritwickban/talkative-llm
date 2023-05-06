import inspect
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Dict, List

import cohere
import openai
import torch
import transformers
from peft import PeftModel
from rich.console import Console
from transformers import AutoTokenizer, GenerationConfig, LlamaTokenizer

console = Console()
error_console = Console(stderr=True, style='bold red')


class LLMCaller(ABC):
    @abstractmethod
    def __init__(self) -> None:
        console.log(f'{self.__class__.__name__} is instantiated.')

    @abstractmethod
    def generate(self, inputs: List[str] | List[Dict]) -> List[Dict] | Dict:
        '''This method passes inputs to either LLM directly or via OpenAI API and retrieves generated results.

        Args:
            inputs: a list of string prompts or a list of of dict messages in chat format as specified by OpenAI.

        Returns: a list of dict containing corresponding generated results or a single dict result in the case of `chat` mode.
        # TODO: elaborate more on the results depending on the two cases.
        '''
        pass

    def update_caller_params(self, new_caller_params: Dict) -> None:
        for param_key, param_value in new_caller_params.items():
            if param_key in self.caller_params:
                self.caller_params[param_key] = param_value


class OpenAICaller(LLMCaller):
    mode_to_api_caller = {
        'chat': openai.ChatCompletion,
        'completion': openai.Completion,
        'edit': openai.Edit,
        'embedding': openai.Embedding,
    }

    def __init__(self, config: Dict) -> None:
        super().__init__()
        assert config['framework'] == 'openai'

        openai.organization = config['organization_id']
        openai.api_key = config['openai_api_key']

        self.mode = config['mode']

        if self.mode not in OpenAICaller.mode_to_api_caller:
            error_console.log(f'Unsupported mode: {self.mode}')
            sys.exit(1)
        self.caller = OpenAICaller.mode_to_api_caller[self.mode]
        self.caller_params = config['params']

        console.log(f'API parameters are:')
        console.log(self.caller_params)

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


class HuggingFaceCaller(LLMCaller):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        assert config['framework'] == 'huggingface'
        self.skip_special_tokens = config['skip_special_tokens']
        self.caller_params = config['params']

        model_type = getattr(transformers, config['mode'])
        model_name = config['model']

        try:
            self.generation_config, unused_kwargs = GenerationConfig.from_pretrained(model_name, **self.caller_params, return_unused_kwargs=True)
            if len(unused_kwargs) > 0:
                console.log('Following config parameters are ignored, please check:')
                console.log(unused_kwargs)
        except OSError:
            error_console.log(f'`generation_config.json` could not be found at https://huggingface.co/{model_name}')
            # TODO: Need to check if just passing self.caller_params are ok for the generate method.
            self.generation_config = GenerationConfig(**self.caller_params)

        self.model = model_type.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        console.log(f'Loaded parameters are:')
        console.log(self.generation_config)

    def generate(self, inputs: List[str] | List[Dict]) -> List[Dict]:
        tokenized_inputs = self.tokenizer(inputs, return_tensors='pt')
        generate_args = set(inspect.signature(self.model.forward).parameters)
        # Remove unused args
        unused_args = [key for key in tokenized_inputs.keys() if key not in generate_args]
        for key in unused_args:
            del tokenized_inputs[key]

        outputs = self.model.generate(**tokenized_inputs, generation_config=self.generation_config)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=self.skip_special_tokens)
        all_results = []
        for decoded_output in decoded_outputs:
            result = {'generation': decoded_output}
            all_results.append(result)
        return all_results


class LLaMACaller(LLMCaller):
    # TODO: Need to incorporate codes from: https://github.com/facebookresearch/llama
    def __init__(self, config: Dict) -> None:
        super().__init__()
        raise NotImplementedError(f'{self.__class__.__name__} is not implemented.')


class AlpacaLoraCaller(LLMCaller):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        assert config['framework'] == 'alpaca-lora'
        assert config['device'] in ['cpu', 'cuda']
        self.device = config['device']
        if self.device == 'cuda':
            assert torch.cuda.is_available(), 'cuda is not available'
        self.lora_weights = config['lora_weights']
        self.load_8bit = config['load_8bit']
        self.skip_special_tokens = config['skip_special_tokens']
        self.caller_params = config['params']


        model_type = getattr(transformers, config['mode'])
        model_name = config['model']

        try:
            self.generation_config, unused_kwargs = GenerationConfig.from_pretrained(model_name, **self.caller_params, return_unused_kwargs=True)
            if len(unused_kwargs) > 0:
                console.log('Following config parameters are ignored, please check:')
                console.log(unused_kwargs)
        except OSError:
            error_console.log(f'`generation_config.json` could not be found at https://huggingface.co/{model_name}')
            # TODO: Need to check if just passing self.caller_params are ok for the generate method.
            self.generation_config = GenerationConfig(**self.caller_params)

        # Call a model depending on using gpu
        if self.device == 'cuda':
            model = model_type.from_pretrained(model_name, load_in_8bit=self.load_8bit, torch_dtype=torch.float16, device_map='auto')
            self.model = PeftModel.from_pretrained(model, self.lora_weights, torch_dtype=torch.float16)
        else:
            model = model_type.from_pretrained(model_name, device_map={'': self.device}, low_cpu_mem_usage=True)
            self.model = PeftModel.from_pretrained(model, self.lora_weights, device_map={'': self.device})

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = 0

        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not self.load_8bit:
            model.half()

        model.eval()

        console.log(f'API parameters are:')
        console.log(self.generation_config)

    def generate(self, inputs: List[str] | List[Dict]) -> List[Dict] | Dict:
        tokenized_inputs = self.tokenizer(inputs, return_tensors='pt')
        tokenized_input_ids = tokenized_inputs['input_ids'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=tokenized_input_ids, generation_config=self.generation_config)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=self.skip_special_tokens)
        all_results = []
        for decoded_output in decoded_outputs:
            result = {'generation': decoded_output}
            all_results.append(result)
        return all_results


class CohereCaller(LLMCaller):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        assert config['framework'] == 'cohere'
        self.api_key = config['cohere_api_key']     # Your Cohere API key
        self.caller = cohere.Client(self.api_key)
        self.caller_params = config['params']

        console.log(f'API parameters are:')
        console.log(self.caller_params)

    def generate(self, inputs: List[str]) -> List[Dict]:
        assert isinstance(inputs, list) and isinstance(inputs[0], str)

        responses = self.caller.batch_generate(prompts=inputs, **self.caller_params)
        all_results = []
        for response in responses:
            for generation in response.generations:
                result = {'generation': generation.text}
                all_results.append(result)
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
    else:
        error_console.log(f'Unsupported framework: {framework}')
        sys.exit(1)
