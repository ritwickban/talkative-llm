import sys
import time
from abc import ABC, abstractmethod
from typing import Dict, List

import openai
from rich.console import Console

from talkative_llm.utils import chunk_with_size_n

console = Console()
error_console = Console(stderr=True, style="bold red")


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
                all_results.append({'generation': choice.text, 'finish_reason': choice.finish_reason})
            return all_results


class HuggingFaceCaller(LLMCaller):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        assert config['framework'] == 'huggingface'


class LLaMACaller(LLMCaller):
    # TODO: Need to incorporate codes from: https://github.com/facebookresearch/llama
    def __init__(self, config: Dict) -> None:
        super().__init__()


def get_supported_llm(config: Dict) -> LLMCaller:
    framework = config['framework']
    if  framework == "openai":
        return OpenAICaller(config)
    elif framework == "huggingface":
        return HuggingFaceCaller(config)
    else:
        error_console.log(f'Unsupported framework: {framework}')
        sys.exit(1)
