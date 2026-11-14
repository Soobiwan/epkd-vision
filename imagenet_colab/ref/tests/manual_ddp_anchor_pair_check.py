import os, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
def net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3,8,3,padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8,5))
def data(rank, it):
    g=torch.Generator().manual_seed(100*it+rank)
    return torch.randn(4,3,8,8,generator=g), torch.randn(3,3,16,16,generator=g)
def run(rank, world, q):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT='29583')
    dist.init_process_group('gloo', rank=rank, world_size=world)
    m=net()
    class Pair(nn.Module):
        def __init__(s,st): super().__init__(); s.student=st
        def forward(s,x,xa=None): return s.student(x) if xa is None else (s.student(x), s.student(xa))
    d=DDP(Pair(m), static_graph=False)
    for it in range(3):
        m.zero_grad()
        for mb in range(2):
            x,xa=data(rank,10*it+mb)
            ctx = d.no_sync() if mb==0 else __import__('contextlib').nullcontext()
            with ctx:
                a,b=d(x,xa); loss=a.pow(2).mean()+0.7*b.pow(2).mean()
                loss.backward()
        if rank==0: q.put([p.grad.detach().numpy().copy() for p in m.parameters()])
    dist.destroy_process_group()
if __name__=='__main__':
    world=2; ctx=mp.get_context('spawn'); q=ctx.Queue()
    ps=[ctx.Process(target=run,args=(r,world,q)) for r in range(world)]
    [p.start() for p in ps]; got=[q.get(timeout=240) for _ in range(3)]; [p.join() for p in ps]
    # reference: no DDP, average of per-rank grads (BN in train mode is per-rank, same as DDP without SyncBN)
    ok=True
    for it in range(3):
        acc=None
        for r in range(world):
            m=net()
            for mb in range(2):
                x,xa=data(r,10*it+mb); (m(x).pow(2).mean()+0.7*m(xa).pow(2).mean()).backward()
            g=[p.grad.clone()/world for p in m.parameters()]
            acc=g if acc is None else [a+b for a,b in zip(acc,g)]
        err=max(abs(a.numpy()-b).max() for a,b in zip(acc,got[it])); print('iter',it,'max grad err',err); ok&=err<1e-5
    print('DDP double-forward OK' if ok else 'MISMATCH')
