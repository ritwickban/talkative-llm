import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, prompts):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    print(f"Running on device {device}")
    
    generation_config = GenerationConfig(  
        max_new_tokens=64,
        early_stopping=True,
        num_beams=3,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        num_return_sequences=1,
    )

    tokenizer = AutoTokenizer.from_pretrained("saltlux/luxia-21.4b-alignment-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "saltlux/luxia-21.4b-alignment-v1.0",
        device_map="auto",
        torch_dtype=torch.float16,
    ).to(device)
    
    # Wrap the model in DDP
    model = DDP(model, device_ids=[rank])

    tokenized_inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.module.generate(**tokenized_inputs, generation_config=generation_config)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    all_results = [{'generation': decoded_output} for decoded_output in decoded_outputs]
    print(all_results)
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    prompts = ['Who won the world series in 2020?',
               'Knock, Knock. Who\'s there?',
               'What makes a great dish?']
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Using torch.multiprocessing.spawn to launch multiple processes
    torch.multiprocessing.spawn(main,
                                args=(world_size, prompts),
                                nprocs=world_size,
                                join=True)
