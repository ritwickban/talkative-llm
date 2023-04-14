import json
import os
import sys
import time
from argparse import Namespace

import yaml
from rich.console import Console
from rich.progress import track

from talkative_llm.llm import get_supported_llm
from talkative_llm.utils import chunk_with_size_n, read_lines, write_lines

console = Console()
error_console = Console(stderr=True, style="bold red")


def launch(args: Namespace) -> None:
    config_path = args.config
    delay_in_seconds = args.delay_in_seconds
    prompt = args.prompt
    output_path = args.output

    # load model config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if os.path.isfile(prompt):
        prompts = read_lines(prompt)
        console.log(f'{len(prompts)} prompts are read from {prompt}:')
    else:
        prompts = [prompt]

    # each line is a list of messages in chat format, e.g. [{"role": "user", "content": "Hello!"}, ..]
    if config['mode'] == 'chat':
        prompts = [json.loads(line) for line in prompts]
        if not isinstance(prompts[0], list) or not isinstance(prompts[0][0], dict):
            error_console.log('For `chat` mode using `OpenAICaller`, each line in an input file must be a list of chat messages (dict).')
            sys.exit(1)

    console.log('Loaded prompts are:')
    console.log(prompts[:3])

    if output_path is not None:
        assert os.path.isdir(os.path.dirname(output_path))

    llm = get_supported_llm(config)

    if config['mode'] == 'chat':
        all_results = []
        for messages in track(prompts, description="Generating..."):
            result = llm.generate(messages)
            all_results.append(result)
    else:
        all_results = []
        if len(prompts) <= config['batch_size']:
            all_results = llm.generate(prompts)
        else:
            for prompt_batch in track(chunk_with_size_n(prompts, config['batch_size']), description="Generating..."):
                results = llm.generate(prompt_batch)
                all_results.extend(results)
                if config['framework'] == 'openai':
                    time.sleep(delay_in_seconds)

    # save results
    if output_path is None:
        console.log(all_results)
    else:
        results_in_str = [json.dumps(result) for result in all_results]
        write_lines(results_in_str, output_path)

    console.log('Done.')