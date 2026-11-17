"""
metrics.py

Measure and compare the performance 
"""

import os
import threading
import time

import numpy as np
import psutil

class Measure:
    """
    Measure execution time and memory for 
    one pipline stage
    """
    def __init__(self, stage="undefined", interval=0.005):
        self.stage = stage
        self.interval = interval
        self.process = psutil.Process(os.getpid())

    def _sample(self):
        while not self._stop.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
                self._stop.wait(self.interval)
            except psutil.Error as e:
                print(f'Error Occurred: {e}')
                continue

    def __enter__(self):
        self.base_rss = self.peak_rss = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def measure_size(self, obj):
        self.size_bytes = len(obj)
        return self.size_bytes

    def summary(self):
        result = {
            "latency_s": self.seconds,
            "peak_rss_mib": self.peak_rss / 2 ** 20,
            "peak_rss_increase_mib": (self.peak_rss - self.base_rss) / 2 ** 20,
        }
        if hasattr(self, "size_bytes"):
            result["size_bytes"] = self.size_bytes
        return {self.stage: result}

def package_nbytes(package):
    return sum(len(blob) for chunks in package["columns"].values() for blob in chunks)


def compare(raw, he):
    keys = ["stat", "target", "group_var", "group"]
    merged = raw.merge(he, on=keys, suffixes=("_raw", "_he"))
    merged["abs_err"] = (merged["value_raw"] - merged["value_he"]).abs()
    merged["rel_err"] = merged["abs_err"] / merged["value_raw"].abs().replace(0, np.nan)
    return merged
