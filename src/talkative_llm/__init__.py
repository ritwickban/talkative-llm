import argparse
from importlib.metadata import metadata

import talkative_llm.cli


def main(*args: str) -> None:
    parser = argparse.ArgumentParser(description='A Python library to query large language models')
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {metadata(__package__)["version"]}')
    parser.add_argument('-c', '--config', type=str, required=True, help='config file for language model to be called')
    parser.add_argument('-o', '--output', type=str, help='path to output file, if not set, print to stdout')
    parser.add_argument('-p', '--prompt', required=True, help='either path to an input prompt JSON file or a single prompt string via stdin')
    parser.add_argument('--delay-in-seconds', type=float, default=3.0, help='delay in seconds for each OpenAI API call')
    parser.set_defaults(func=talkative_llm.cli.launch)
    args = parser.parse_args()
    args.func(args)