from abc import ABC, abstractmethod
from ast import Dict

from rich.console import Console

console = Console()


def get_supported_llm(model_type: str, **kwargs: Dict) -> bool:
    llm_classes = [GPT2Caller, GPT3Caller, LlamaCaller, ChatGPTCaller]
    for _class in llm_classes:
        if model_type in _class.aliases:
            return _class(**kwargs)
    raise TypeError(f'{model_type} is not supported.')


class LlmCaller(ABC):
    @abstractmethod
    def __init__(self) -> None:
        console.log(f'{self.__class__.__name__} is instantiated.')


class GPT2Caller(LlmCaller):
    aliases = ['gpt2', 'gpt-2', 'GPT2', 'GPT-2']

    def __init__(self) -> None:
        super().__init__()
        pass


class GPT3Caller(LlmCaller):
    aliases = ['gpt3', 'gpt-3', 'GPT3', 'GPT-3']

    def __init__(self) -> None:
        super().__init__()
        pass


class LlamaCaller(LlmCaller):
    aliases = ['llama', 'Llama', 'LLAMA']

    def __init__(self) -> None:
        super().__init__()
        pass


class ChatGPTCaller(LlmCaller):
    aliases = ['chatgpt', 'Chatgpt', 'ChatGpt', 'ChatGPT']

    def __init__(self) -> None:
        super().__init__()
        pass
