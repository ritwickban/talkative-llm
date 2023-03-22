# talkative-llm
- A library to query large language models (LLMs) given a file of prompts.
- Supported LLMs include:
  - Via HuggingFace models: GPT-2, FLAN-T5, FLAN-UL2
  - Via OpenAI's API: GPT-3, ChatGPT (GPT-3.5), GPT-4
  - Llama (need to download weights from [Meta](https://github.com/facebookresearch/llama))
- Clone, (make venv), and do `pip install -e .` for editable installation.
  - You may want to `git checkout` and use `develop` branch for cutting-edge features.

### Development process
- Push to `develop` branch directly if your changes don't require code reviews. Otherwise, create a new branch and do pull-requests to the `develop` branch.
- Once enough features are implemented to `develop` branch, it will be peer-reviewd and merged to `main` branch.

```
Usage: talkative_llm [-h] [-v] -c CONFIG -p PROMPT [-o OUTPUT] [--delay-in-seconds DELAY_IN_SECONDS]

Python library for querying large language models

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -c CONFIG, --config CONFIG
                        config file for language model to be called
  -p PROMPT, --prompt PROMPT
                        either path to an input prompt file or a single prompt string
  -o OUTPUT, --output OUTPUT
                        path to output file, if not set, print to stdout
  --delay-in-seconds DELAY_IN_SECONDS
                        delay in seconds for each OpenAI API call
```

### OpenAI's Completion (GPT3)
<img width="1276" alt="image" src="https://user-images.githubusercontent.com/3746478/226238549-14f01831-d709-4657-bf9a-749774a31730.png">

### OpenAI's ChatCompletion (ChatGPT)
<img width="1279" alt="image" src="https://user-images.githubusercontent.com/3746478/226238673-840d4cfa-b26b-449b-abd3-659e2e3365d9.png">
