import os
from argparse import Namespace

from rich.console import Console
from rich.progress import track

from talkative_llm.llm import get_supported_llm
from talkative_llm.utils import read_lines

console = Console()


def launch(args: Namespace) -> None:
    model_type = args.model
    prompt = args.prompt
    output_path = args.output

    if os.path.isfile(prompt):
        prompts = read_lines(prompt)
        console.log(f'{len(prompts)} prompts are read from {prompt}:')
        console.log(f'\n'.join(prompts[:5] + ['...']))
    else:
        console.log(f'prompt is:\n{prompt}')
        prompts = [prompt]

    if output_path is not None:
        assert os.path.isdir(os.path.dirname(output_path))

    # check model_type TODO: pass kwargs
    llm = get_supported_llm(model_type)

    # batch loop through generation

    # save results

    console.log('Done.')
