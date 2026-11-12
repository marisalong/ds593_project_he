"""
config.py

Holds configurations for the he pipeline

"""
# TODO: we can chnage this to use pydantic or dataclasses for enforcement

# I also think we need to explain better what these variables mean

# Categorical variables and how they should be encoded
CATEGORY_DOMAINS = {
    "INCOME_BRACKET": [0, 1],          # 0: <= $50k, 1: > $50k
    "SCHL": list(range(0, 25)),        # educational attainment
    "RAC1P": list(range(1, 10)),       # race
    "SEX": [1, 2],                     # sex
    "COW": list(range(0, 10)),         # class of worker
    "MAR": list(range(1, 6)),          # marital status
}
GROUP_VARS = ["INCOME_BRACKET", "SCHL", "RAC1P", "SEX", "COW"]

# Variables for which you want to calculate the mean
MEAN_TARGETS = ["PINCP", "AGEP", "WKHP"]