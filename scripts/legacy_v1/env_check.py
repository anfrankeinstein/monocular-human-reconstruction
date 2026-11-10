"""环境自检：Python / torch / CUDA / 显存 / 关键依赖。"""
import sys

print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__)
    print("torch cuda build", torch.version.cuda)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))
        free, total = torch.cuda.mem_get_info()
        print(f"vram {free/1024**3:.1f}G free / {total/1024**3:.1f}G total")
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()
        print("bf16 matmul ok", tuple(y.shape))
except Exception as e:
    print("torch check failed:", repr(e))

for mod in ["numpy", "cv2", "mediapipe", "trimesh", "pygltflib"]:
    try:
        m = __import__(mod)
        print(f"{mod} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"{mod} MISSING ({type(e).__name__})")
