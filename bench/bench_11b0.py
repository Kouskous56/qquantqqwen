"""11B0: torch sparse semi-structured feasibility + microbench (Qwen-3B shapes)."""
import torch, time
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("arch:", torch.cuda.get_device_capability(0))
try:
    from torch.sparse import to_sparse_semi_structured
    print("sparse API: available")
except Exception as e:
    print("sparse API MISSING:", e)
    raise SystemExit

SHAPES = {"q/o 2048x2048": (2048, 2048), "k/v 2048x256": (2048, 256),
          "gate/up 2048x11008": (2048, 11008), "down 11008x2048": (11008, 2048)}

def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000

for name, (out_f, in_f) in SHAPES.items():
    W = torch.randn(out_f, in_f, dtype=torch.float16, device="cuda")
    # tao mask 2:4 hop le: moi nhom 4 giu 2 |W| lon nhat
    g = W.abs().view(-1, in_f // 4, 4)
    _, idx = torch.topk(g, k=2, dim=-1)
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True)
    W24 = torch.where(keep.view_as(W), W, torch.zeros_like(W))
    try:
        S = to_sparse_semi_structured(W24)
    except Exception as e:
        print(f"{name}: compress FAILED: {type(e).__name__} {str(e)[:120]}")
        continue
    X = torch.randn(512, in_f, dtype=torch.float16, device="cuda")
    try:
        t_dense = bench(lambda: X @ W24.t())
        t_sparse = bench(lambda: torch.sparse.mm(S, X.t()).t())
        # verify
        err = ((X @ W24.t()) - torch.sparse.mm(S, X.t()).t()).abs().max().item()
        print(f"{name}: dense {t_dense:.2f}ms sparse {t_sparse:.2f}ms "
              f"speedup {t_dense/t_sparse:.2f}x maxerr {err:.2e}", flush=True)
    except Exception as e:
        print(f"{name}: mm FAILED: {type(e).__name__} {str(e)[:150]}")
