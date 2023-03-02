import argparse

import talkative_llm.cli
import talkative_llm.meta_data


def main(*args: str) -> None:
    parser = argparse.ArgumentParser(description=talkative_llm.meta_data.description)
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {talkative_llm.meta_data.version}')
    parser.add_argument('-m', '--model', type=str, required=True, help='name of large language model')
    parser.add_argument('-p', '--prompt', required=True, help='either path to an input prompt file or a single prompt string')
    parser.add_argument('-o', '--output', type=str, help='path to output file, if not set, print to stdout')
    parser.set_defaults(func=talkative_llm.cli.launch)
    args = parser.parse_args()
    args.func(args)