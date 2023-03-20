# talkative-llm
- A library to query large language models (LLMs) given a file of prompts.
- (To be) supported LLMs include:
  - GPT-2 (huggingface), GPT-3 (OpenAI API), Llama (downloaded weights from [Meta](https://github.com/facebookresearch/llama)), ChatGPT (OpenAI API)
```
sage: talkative_llm [-h] [-v] -c CONFIG -p PROMPT [-o OUTPUT] [--delay-in-seconds DELAY_IN_SECONDS]

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
