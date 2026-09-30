"""Separate process RSS, sampled Metal allocations, and CUDA allocator peaks."""

import threading

import psutil


class MemoryProbe:
    def __init__(self, torch, device, interval=0.02):
        self.torch, self.device, self.interval = torch, device, interval
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.rss = self.allocated = self.driver = self.samples = 0

    def sample(self):
        self.rss = max(self.rss, self.process.memory_info().rss)
        if self.device == "mps":
            self.allocated = max(self.allocated, self.torch.mps.current_allocated_memory())
            self.driver = max(self.driver, self.torch.mps.driver_allocated_memory())
        self.samples += 1

    def run(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        return {
            "sampling_interval_seconds": self.interval,
            "samples": self.samples,
            "sampled_peak_process_rss_bytes": self.rss,
            "sampled_peak_mps_allocated_bytes": self.allocated if self.device == "mps" else None,
            "sampled_peak_mps_driver_bytes": self.driver if self.device == "mps" else None,
            "cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated()
            if self.device == "cuda"
            else None,
            "limitation": "RSS and Metal peaks are 20ms samples; CUDA is PyTorch allocator peak. Not total system/GPU memory; ZeroGPU packed storage can lie outside the allocator. Sampling adds measurement overhead.",
        }
