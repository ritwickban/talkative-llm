import argparse

import yaml
from rich.console import Console

console = Console()
error_console = Console(stderr=True, style='bold red')


def launch(args: argparse.Namespace) -> None:
    config_path = args.config
    output_path = args.output
    prompt = args.prompt
    delay_in_seconds = args.delay_in_seconds

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    console.log(config)