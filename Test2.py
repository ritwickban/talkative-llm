import torch
from accelerate import Accelerator, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# Initialize the Accelerator
accelerator = Accelerator()
torch.cuda.empty_cache()
generation_config = GenerationConfig(
    max_new_tokens=64,
    early_stopping=True,
    num_beams=3,
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    num_return_sequences=1,
)

PROMPTS = [
    'Who won the world series in 2020?',
    'Knock, Knock. Who\'s there?',
    'What makes a great dish?'
]
#if not torch.distributed.is_initialized():
#    torch.distributed.init_process_group(backend='nccl')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("saltlux/luxia-21.4b-alignment-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "saltlux/luxia-21.4b-alignment-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)

#local_folder='/panfs/jay/groups/29/dongyeop/baner212/NLPInternship/checkpoints/'
#model.save_pretrained(local_folder)
#tokenizer.save_pretrained(local_folder)

#model = load_checkpoint_and_dispatch(
#    model=AutoModelForCausalLM.from_pretrained("saltlux/luxia-21.4b-alignment-v1.0"),
#    torch_dtype=torch.float16,
#    device_map="auto"
#)
#tokenizer.to(accelerator.device)
#tokenizer = AutoTokenizer.from_pretrained(local_folder)

#model.to(accelerator.device)

#model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[accelerator.local_process_index])

# Prepare the model and data with the accelerator
model,tokenizer = accelerator.prepare(model,tokenizer)

# Tokenizing the inputs
tokenized_inputs = tokenizer(PROMPTS, return_tensors='pt', padding=True).to(accelerator.device)

# Generate the outputs
with torch.no_grad():
    outputs = model.generate(**tokenized_inputs, generation_config=generation_config)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    all_results=[]
    for decoded_output in decoded_outputs:
            all_results.append({'generation': decoded_output})

    torch.cuda.empty_cache() #Clearing CUDA Cache



print(all_results)
