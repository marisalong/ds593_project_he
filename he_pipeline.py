import os
import threading
import time

import numpy as np
import pandas as pd
import psutil
import tenseal as ts

CATEGORY_DOMAINS = {
    "INCOME_BRACKET": [0, 1],          # 0: <= $50k, 1: > $50k
    "SCHL": list(range(0, 25)),        # educational attainment
    "RAC1P": list(range(1, 10)),       # race
    "SEX": [1, 2],                     # sex
    "COW": list(range(0, 10)),         # class of worker
    "MAR": list(range(1, 6)),          # marital status
}
GROUP_VARS = ["INCOME_BRACKET", "SCHL", "RAC1P", "SEX", "COW"]
MEAN_TARGETS = ["PINCP", "AGEP", "WKHP"]


def _indicator(var, level):
    return f"{var}={level}"

def prepare(df):
    out = df[["AGEP", "WKHP", "PINCP", "SCHL", "RAC1P", "SEX", "COW", "MAR"]].copy()
    out["INCOME_BRACKET"] = (df["PINCP"] > 50_000).astype(int)
    for var in CATEGORY_DOMAINS:
        out[var] = out[var].astype(int)
    return out.reset_index(drop=True)


def features_plain(df):
    df = prepare(df)
    rows = []
    n = len(df)
    for target in MEAN_TARGETS:
        rows.append(("mean", target, "ALL", "ALL", df[target].mean()))
    for var in GROUP_VARS:
        for level in CATEGORY_DOMAINS[var]:
            mask = df[var] == level
            rows.append(("count", "n", var, level, float(mask.sum())))
            for target in MEAN_TARGETS:
                value = df.loc[mask, target].mean() if mask.any() else np.nan
                rows.append(("mean", target, var, level, value))
    rows.append(("corr", "AGEP~PINCP", "ALL", "ALL", df["AGEP"].corr(df["PINCP"])))
    for level in CATEGORY_DOMAINS["MAR"]:
        indicator = (df["MAR"] == level).astype(float)
        value = df["WKHP"].corr(indicator) if 0 < indicator.sum() < n else np.nan
        rows.append(("corr", "WKHP~MAR", "MAR", level, value))
    return pd.DataFrame(rows, columns=["stat", "target", "group_var", "group", "value"])


class DataOwner:
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60),
                 global_scale=2 ** 40):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
        )
        self.context.global_scale = global_scale
        # Galois keys are needed for rotations in .sum() / .dot().
        self.context.generate_galois_keys()
        self.slots = poly_modulus_degree // 2

    def public_context(self):
        return self.context.serialize(save_secret_key=False)

    def encode(self, df):
        df = prepare(df)
        columns = {target: df[target].astype(float).to_numpy() for target in MEAN_TARGETS}
        for var, levels in CATEGORY_DOMAINS.items():
            for level in levels:
                columns[_indicator(var, level)] = (df[var] == level).astype(float).to_numpy()
        return pd.DataFrame(columns)

    def encrypt(self, encoded):
        n = len(encoded)
        package = {"n_rows": n, "columns": {}}
        for name in encoded.columns:
            values = encoded[name].to_numpy()
            package["columns"][name] = [
                ts.ckks_vector(self.context, values[i:i + self.slots].tolist()).serialize()
                for i in range(0, n, self.slots)
            ]
        return package

    def decrypt(self, encrypted_stats):
        return {
            key: ts.ckks_vector_from(self.context, blob).decrypt()[0]
            for key, blob in encrypted_stats.items()
        }

    @staticmethod
    def finalize(stats, n):
        rows = []
        for target in MEAN_TARGETS:
            rows.append(("mean", target, "ALL", "ALL", stats[f"sum[{target}]"] / n))
        for var in GROUP_VARS:
            for level in CATEGORY_DOMAINS[var]:
                ind = _indicator(var, level)
                count = stats[f"sum[{ind}]"]
                # CKKS is approximate: a count of 0 decrypts to ~1e-6, not 0.
                empty = round(count) == 0
                rows.append(("count", "n", var, level, count))
                for target in MEAN_TARGETS:
                    value = np.nan if empty else stats[f"dot[{target},{ind}]"] / count
                    rows.append(("mean", target, var, level, value))

        def pearson(sx, sy, sxx, syy, sxy):
            cov = n * sxy - sx * sy
            var_x, var_y = n * sxx - sx ** 2, n * syy - sy ** 2
            return cov / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else np.nan

        rows.append(("corr", "AGEP~PINCP", "ALL", "ALL", pearson(
            stats["sum[AGEP]"], stats["sum[PINCP]"],
            stats["dot[AGEP,AGEP]"], stats["dot[PINCP,PINCP]"], stats["dot[AGEP,PINCP]"],
        )))
        for level in CATEGORY_DOMAINS["MAR"]:
            ind = _indicator("MAR", level)
            s_ind = stats[f"sum[{ind}]"]
            # indicator^2 == indicator, so sum(ind^2) = sum(ind).
            value = np.nan if round(s_ind) in (0, n) else pearson(
                stats["sum[WKHP]"], s_ind,
                stats["dot[WKHP,WKHP]"], s_ind, stats[f"dot[WKHP,{ind}]"],
            )
            rows.append(("corr", "WKHP~MAR", "MAR", level, value))
        return pd.DataFrame(rows, columns=["stat", "target", "group_var", "group", "value"])


class UntrustedResearcher:
    def __init__(self, public_context):
        self.context = ts.context_from(public_context)
        assert not self.context.has_secret_key(), "researcher must not hold the secret key"

    def _load(self, package):
        return {
            name: [ts.ckks_vector_from(self.context, blob) for blob in chunks]
            for name, chunks in package["columns"].items()
        }

    @staticmethod
    def _sum(chunks):
        total = chunks[0].sum()
        for chunk in chunks[1:]:
            total += chunk.sum()
        return total

    @staticmethod
    def _dot(a_chunks, b_chunks):
        total = a_chunks[0].dot(b_chunks[0])
        for a, b in zip(a_chunks[1:], b_chunks[1:]):
            total += a.dot(b)
        return total

    def required_statistics(self):
        sums = list(MEAN_TARGETS)
        dots = [("AGEP", "AGEP"), ("PINCP", "PINCP"), ("AGEP", "PINCP"), ("WKHP", "WKHP")]
        for var in GROUP_VARS:
            for level in CATEGORY_DOMAINS[var]:
                ind = _indicator(var, level)
                sums.append(ind)
                dots += [(target, ind) for target in MEAN_TARGETS]
        for level in CATEGORY_DOMAINS["MAR"]:
            ind = _indicator("MAR", level)
            sums.append(ind)
            dots.append(("WKHP", ind))
        return sums, list(dict.fromkeys(dots))

    def compute(self, package):
        cols = self._load(package)
        sums, dots = self.required_statistics()
        out = {}
        for name in sums:
            out[f"sum[{name}]"] = self._sum(cols[name]).serialize()
        for a, b in dots:
            out[f"dot[{a},{b}]"] = self._dot(cols[a], cols[b]).serialize()
        return out

class Measure:
    def __init__(self, interval=0.005):
        self.interval = interval
        self.process = psutil.Process(os.getpid())

    def _sample(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self._stop.wait(self.interval)

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

    def summary(self):
        return {
            "latency_s": self.seconds,
            "peak_rss_mib": self.peak_rss / 2 ** 20,
            "peak_rss_increase_mib": (self.peak_rss - self.base_rss) / 2 ** 20,
        }


def package_nbytes(package):
    return sum(len(blob) for chunks in package["columns"].values() for blob in chunks)


def compare(raw, he):
    keys = ["stat", "target", "group_var", "group"]
    merged = raw.merge(he, on=keys, suffixes=("_raw", "_he"))
    merged["abs_err"] = (merged["value_raw"] - merged["value_he"]).abs()
    merged["rel_err"] = merged["abs_err"] / merged["value_raw"].abs().replace(0, np.nan)
    return merged
