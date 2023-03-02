# talkative-llm
- A library to query large language models (LLMs) given a file of prompts.
- (To be) supported LLMs include:
  - GPT-2 (huggingface), GPT-3 (OpenAI API), Llama (downloaded weights from [Meta](https://github.com/facebookresearch/llama)), ChatGPT (OpenAI API)
```
usage: talkative_llm [-h] [-v] -m MODEL -p PROMPT [-o OUTPUT]

Python library for querying large language models

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -m MODEL, --model MODEL
                        name of large language model
  -p PROMPT, --prompt PROMPT
                        either path to an input prompt file or a single prompt string
  -o OUTPUT, --output OUTPUT
                        path to output file, if not set, print to stdout
```
