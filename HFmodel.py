import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



PROMPTS = ['Who won the world series in 2020?',
           'Knock, Knock. Who\'s there?',
           'What makes a great dish?']

tokenizer = AutoTokenizer.from_pretrained("saltlux/luxia-21.4b-alignment-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "saltlux/luxia-21.4b-alignment-v0.1",
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenized_inputs = self.tokenizer(PROMPTS, return_tensors='pt', padding=True).to(self.device)
with torch.no_grad():
        outputs = self.model.generate(**tokenized_inputs, generation_config=self.generation_config)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=self.skip_special_tokens)
        all_results = []
        for decoded_output in decoded_outputs:
            all_results.append({'generation': decoded_output})

print(all_results)