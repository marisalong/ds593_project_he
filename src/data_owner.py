import numpy as np
import pandas as pd
import tenseal as ts

# import manual configurations 
import config.config as config


def _indicator(var, level):
    return f"{var}={level}"

def validate_prepared(df, category_domains, target):
    """
    This method ensures that the dataframe has been adequately
    preprocessed for the HE pipeline
    """
    for category in category_domains:
        if category not in df.columns:
            raise IndexError(f'{category} not found in the dataframe columns.')
    if target not in df.columns:
        raise IndexError(f'{category} not found in the dataframe columns.')
    if df.isna().values.any():
        raise ValueError('dataframe contains nan values')
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        raise TypeError(f'dataframe contains non-numeric values.')
    out = df[category_domains + [target]].copy()
    return out.reset_index(drop=True)

class DataOwner:
    """
    Class represeting the data owner.
    
    Includes methods:
        public_context: 
        encode:
        encrypt: perform homomorphic encryption

    """
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60),
                 global_scale=2 ** 40, mean_targets= config.MEAN_TARGETS, category_domains=config.CATEGORY_DOMAINS):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
        )
        self.context.global_scale = global_scale
        # Galois keys are needed for rotations in .sum() / .dot().
        self.context.generate_galois_keys()
        self.slots = poly_modulus_degree // 2
        self.mean_targets = mean_targets ## TODO: consider changing this depending on how we make the functions more flexible
        self.category_domains = category_domains ## TODO: same with this

    def public_context(self):
        return self.context.serialize(save_secret_key=False)

    def encode(self, df):
        """
        prepares for encryption by converting: 
            - MEAN_TARGETS columns to floats and 
            - CATEGORY_DOMAINS to one-hot encoded variables
        """
        try:
            df = validate_prepared(df)
        except Exception as e:
            raise AssertionError(f'dataframe is not valid: {e}')
        columns = {target: df[target].astype(float).to_numpy() for target in self.mean_targets}
        for var, levels in self.category_domains.items():
            for level in levels:
                columns[_indicator(var, level)] = (df[var] == level).astype(float).to_numpy()
        return pd.DataFrame(columns)

    def encrypt(self, encoded):
        """
        Encrypt the encoded values
        """
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
        """
        Decrypt the returned values
        """
        return {
            key: ts.ckks_vector_from(self.context, blob).decrypt()[0]
            for key, blob in encrypted_stats.items()
        }

    @staticmethod
    def finalize(stats, n):
        rows = []
        for target in config.MEAN_TARGETS:
            rows.append(("mean", target, "ALL", "ALL", stats[f"sum[{target}]"] / n))
        for var in config.GROUP_VARS:
            for level in config.CATEGORY_DOMAINS[var]:
                ind = _indicator(var, level)
                count = stats[f"sum[{ind}]"]
                # CKKS is approximate: a count of 0 decrypts to ~1e-6, not 0.
                empty = round(count) == 0
                rows.append(("count", "n", var, level, count))
                for target in config.MEAN_TARGETS:
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
        for level in config.CATEGORY_DOMAINS["MAR"]:
            ind = _indicator("MAR", level)
            s_ind = stats[f"sum[{ind}]"]
            # indicator^2 == indicator, so sum(ind^2) = sum(ind).
            value = np.nan if round(s_ind) in (0, n) else pearson(
                stats["sum[WKHP]"], s_ind,
                stats["dot[WKHP,WKHP]"], s_ind, stats[f"dot[WKHP,{ind}]"],
            )
            rows.append(("corr", "WKHP~MAR", "MAR", level, value))
        return pd.DataFrame(rows, columns=["stat", "target", "group_var", "group", "value"])