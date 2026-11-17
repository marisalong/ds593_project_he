import os
import threading
import time

import numpy as np
import pandas as pd
import psutil
import tenseal as ts

# import manual configurations 
import config.config as config


def _indicator(var, level):
    return f"{var}={level}"


class Collaborator:
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
        sums = list(config.MEAN_TARGETS)
        dots = [("AGEP", "AGEP"), ("PINCP", "PINCP"), ("AGEP", "PINCP"), ("WKHP", "WKHP")]
        for var in config.GROUP_VARS:
            for level in config.CATEGORY_DOMAINS[var]:
                ind = _indicator(var, level)
                sums.append(ind)
                dots += [(target, ind) for target in config.MEAN_TARGETS]
        for level in config.CATEGORY_DOMAINS["MAR"]:
            ind = _indicator("MAR", level)
            sums.append(ind)
            dots.append(("WKHP", ind))
        return sums, list(dict.fromkeys(dots))

    def compute(self, package):
        ## TODO: edit this to be able to take in a method, rather than
        ## calculating just the statstics we defined
        cols = self._load(package)
        sums, dots = self.required_statistics()
        out = {}
        for name in sums:
            out[f"sum[{name}]"] = self._sum(cols[name]).serialize()
        for a, b in dots:
            out[f"dot[{a},{b}]"] = self._dot(cols[a], cols[b]).serialize()
        return out
