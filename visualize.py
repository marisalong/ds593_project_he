"""
visualize.py

holds code for reusable visualization
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmarks(results, experiment_name="unnamed_experiment"):

    stages = [
        "raw_features",
        "initialization", 
        "encryption", 
        "compute_features", 
        "he_decrypt", 
        "finalize"
    ]

    times = [
        results[stage]["latency_s"] for stage in stages
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(stages, times, color="steelblue")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"plots/{experiment_name}_execution_time.png", dpi=300)
    plt.show()

    # 2. Peak memory
    memory = [
        results[stage]["peak_rss_mib"] for stage in stages
    ]
    memory_incr = [
            results[stage]["peak_rss_increase_mib"] for stage in stages
    ]
    x = np.arange(len(stages))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, memory, width, label='Memory', color='darkorange')
    plt.bar(x + width/2, memory_incr, width, label='Memory Increment', color='darkblue')
    plt.ylabel("Peak RSS (MiB) and RSS Increase (MiB)")
    plt.title("Peak Memory Consumption and Memory Increase")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"plots/{experiment_name}memory_usage.png", dpi=300)
    plt.show()

    # 3. Storage size
    # sizes = [
    #     results["public_evaluation_context_size_bytes"],
    #     results["ciphertext_storage_bytes"],
    #     results["encrypted_output_bytes"]
    # ]

    # labels = [
    #     "Public Context",
    #     "Encrypted Input",
    #     "Encrypted Output"
    # ]

    # sizes_mb = np.array(sizes) / (1024 ** 2)

    # plt.figure(figsize=(9, 5))
    # plt.bar(labels, sizes_mb, color="seagreen")
    # plt.ylabel("Size (MiB)")
    # plt.title("Homomorphic Encryption Storage Overhead")
    # plt.tight_layout()
    # plt.savefig(f"plots/{experiment_name}storage_size.png", dpi=300)
    # plt.show()
