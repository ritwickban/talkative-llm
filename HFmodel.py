import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generation_config = GenerationConfig(  
      max_new_tokens= 64,
      early_stopping= True,
      num_beams= 3,
      temperature= 1.0,
      top_p= 1.0,
      top_k= 50,
      num_return_sequences= 1,
      )

PROMPTS = ['Who won the world series in 2020?',
           'Knock, Knock. Who\'s there?',
           'What makes a great dish?']

tokenizer = AutoTokenizer.from_pretrained("saltlux/luxia-21.4b-alignment-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "saltlux/luxia-21.4b-alignment-v1.0",
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenized_inputs = tokenizer(PROMPTS, return_tensors='pt', padding=True).to(device)
with torch.no_grad():
        outputs = model.generate(**tokenized_inputs, generation_config=generation_config)
        decoded_outputs = [tokenizer.batch_decode(outputs, skip_special_tokens=True) for output in outputs]
        all_results = []
        for decoded_output in decoded_outputs:
            all_results.append({'generation': decoded_output})

print(all_results)