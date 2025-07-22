import os
import torch
import torch.distributed as dist


def setup_ddp():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def gather_tensor_across_processes(tensor):
    world_size = dist.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)