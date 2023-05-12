# talkative-llm
- A library to query large language models (LLMs) given a file of prompts.
- Supported LLMs can be found [here](https://docs.google.com/spreadsheets/d/1CSA52gXEOIkmzwj78jT50zQCzIS3uV5gkK66akvBHkk/edit#gid=0).
- Clone, (make venv), and do `pip install -e .` for editable installation.
  - You may want to `git checkout` and use `develop` branch for cutting-edge features.
- LLMs such as LLaMA, Baize, and Vicuna require pre-downloaded weights which can be found at `cs-u-eagle.cs.umn.edu:/home/jonginn/volume/models`
  - Bigger LLaMA weights are `cs-u-elk.cs.umn.edu:/home/zaemyung/Models/LLaMA`
- Before using, create `key.py` for saving all your keys. This file will be ignored by git.
  - Keys to save (Just keep an empty string for unused ones):
    - `COHERE_API_KEY`
    - `OPENAI_API_KEY`
    - `OPENAI_ORGANIZATION_ID`

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
                        either path to an input JSONL prompt file or a single prompt string
  -o OUTPUT, --output OUTPUT
                        path to output file, if not set, print to stdout
  --delay-in-seconds DELAY_IN_SECONDS
                        delay in seconds for each OpenAI API call
```

An example line of an input JSONL file for `chat` mode - each line is a JSON:
```json
[{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "My name is Harry Potter."}, {"role": "assistant", "content": "Hello, there."}, {"role": "user", "content": "What is my name?"}]
```

An example line of an input JSONL file for any other mode - each line is a JSON:
```json
{"prompt": "Tell me a joke."}
```

Run as:
```bash
$ talkative_llm -c openai_gpt-3.5-turbo.yaml -p test_prompt.json -o ./out.json
```

### OpenAI's Completion (GPT3)
<img width="1276" alt="image" src="https://user-images.githubusercontent.com/3746478/226238549-14f01831-d709-4657-bf9a-749774a31730.png">

### OpenAI's ChatCompletion (ChatGPT)
<img width="1279" alt="image" src="https://user-images.githubusercontent.com/3746478/226238673-840d4cfa-b26b-449b-abd3-659e2e3365d9.png">
