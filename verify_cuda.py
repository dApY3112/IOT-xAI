"""verify_cuda.py
Small script to help verify NVIDIA driver, CUDA toolkit, cuDNN, PyTorch and TensorFlow GPU availability.

Usage:
    python verify_cuda.py

It prints `nvidia-smi` info (if available) and checks torch/tensorflow cuda availability.
"""
import shutil
import subprocess
import sys


def run_cmd(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        return f"Error running {cmd}: returncode={e.returncode}\n{e.output}"


def check_nvidia_smi():
    path = shutil.which("nvidia-smi")
    if not path:
        print("nvidia-smi not found on PATH. NVIDIA driver may not be installed or PATH is not set.")
        return
    print("nvidia-smi found at:", path)
    print("--- nvidia-smi output ---")
    print(run_cmd("nvidia-smi -L || nvidia-smi"))


def check_pytorch():
    try:
        import torch
    except Exception as e:
        print("PyTorch not installed or failed to import:", e)
        return
    print("PyTorch version:", torch.__version__)
    try:
        has_cuda = torch.cuda.is_available()
        print("torch.cuda.is_available():", has_cuda)
        if has_cuda:
            print("CUDA devices:")
            for i in range(torch.cuda.device_count()):
                print(f"  [{i}] {torch.cuda.get_device_name(i)}")
            x = torch.randn(3, 3).cuda()
            print("Simple CUDA tensor test OK, tensor device:", x.device)
    except Exception as e:
        print("Error checking PyTorch CUDA status:", e)


def check_tensorflow():
    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow not installed or failed to import:", e)
        return
    print("TensorFlow version:", tf.__version__)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print("TensorFlow found GPUs:", gpus)
        if gpus:
            for g in gpus:
                print("  ", g)
            # quick device placement test
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.linalg.matmul(a, a)
            print("TensorFlow simple GPU matmul result shape:", b.shape)
    except Exception as e:
        print("Error checking TensorFlow GPU status:", e)


def main():
    print("== NVIDIA / CUDA verification script ==")
    check_nvidia_smi()
    print()
    check_pytorch()
    print()
    check_tensorflow()


if __name__ == '__main__':
    main()
