"""
Table-specific metadata for global access.

Todo:
1. Modify to Dict[str, Any] (e.g., {<col>: <dtype>})
"""
# =Raw table=
# ECG
ECG = {
    # Number of samples
    "N_SAMPLES": 13968000,
    # Number of columns
    "N_COLS": 14,
    # Columns
    "COLUMNS": [
        "uid",
        "time_stamp",
        "leadi",
        "leadii",
        "leadiii",
        "leadavr",
        "leadavl",
        "leadavf",
        "leadv1",
        "leadv2",
        "leadv3",
        "leadv4",
        "leadv5",
        "leadv6",
    ],
}
# Amex
Amex = {
    # Number of samples
    "N_SAMPLES": 0,
    # Number of columns
    "N_COLS": 0,
    # Columns
    "COLUMNS": [""],
}
