"""Synthetic data generator.

**Todo**:
- [ ] Consider missing issue
- [ ] Add pre-sorted option
"""
import numpy as np
import pandas as pd


class DataGenerator:
    """Synthetic data generator.

    Following is the data type information of the synthetic data:
    - String identifier column: `str`
    - Integer identifier column: `np.int32`
    - Integer feature column: `np.int32`
    - Floating-point feature column: `np.float32`

    In order to profile ETL operations based on identifier column with
    different cardinalities, high-cardinality identifier column is
    prioritized. That is, if `n_str_ids` or `n_int_ids` is set to `1`,
    the identifier column is generated with high cardinality.

    **Parameters**:
    - `n_samples`: number of samples (*i.e.*, rows)
    - `n_str_ids`: number of string identifiers
    - `n_int_ids`: number of integer identifiers
    - `n_clusts_per_id`: number of clusters in each identifier
    - `n_int_features`: number of integer features
    - `n_float_features`: number of floating-point features
    - `random_state`: control randomness of data generation process
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_str_ids: int = 0,
        n_int_ids: int = 0,
        n_clusts_per_id: int = 0,
        n_int_features: int = 0,
        n_float_features: int = 0,
        random_state: int = 42,
    ):
        self.n_samples = n_samples
        self.n_str_ids = n_str_ids
        self.n_int_ids = n_int_ids
        self.n_clusts_per_id = n_clusts_per_id
        self.n_int_features = n_int_features
        self.n_float_features = n_float_features
        self.random_state = random_state

    def run(self) -> pd.DataFrame:
        """Generate and return the synthetic dataset.

        **Return**:
        - `df`: synthetic dataset
        """
        # Seed generation process to guarantee reproducibility
        np.random.seed(self.random_state)

        # Start generating synthetic dataset
        df = pd.DataFrame()

        for i in range(self.n_str_ids - 1):
            df[f"str_id{i}"] = self._gen_single_id("str")
        if self.n_str_ids != 0:
            df["str_id_hc"] = self._gen_single_id("str", high_cardinality=True)

        for i in range(self.n_int_ids - 1):
            df[f"int_id{i}"] = self._gen_single_id("int")
        if self.n_int_ids != 0:
            df["int_id_hc"] = self._gen_single_id("int", high_cardinality=True)

        for i in range(self.n_int_features):
            df[f"int_f{i}"] = self._gen_single_feature("int")

        for i in range(self.n_float_features):
            df[f"float_f{i}"] = self._gen_single_feature("float")

        return df

    def _gen_single_id(self, dtype: str, high_cardinality: bool = False) -> np.ndarray:
        """Generate a single identifier column.

        Parameters:
            dtype: data type of the feature, either `str` or `int`
            high_cardinality: whether to increase uniqueness of values

        Return:
            id_col: generated identifier column
        """
        if high_cardinality:
            n_clusts_per_id = int(self.n_samples / self.n_clusts_per_id)
        else:
            n_clusts_per_id = self.n_clusts_per_id

        if dtype == "str":
            id_base = np.array([f"c{i}" for i in range(n_clusts_per_id)])
        elif dtype == "int":
            id_base = np.arange(n_clusts_per_id, dtype=np.int32)

        id_col = np.random.choice(id_base, self.n_samples)

        return id_col

    def _gen_single_feature(self, dtype: str) -> np.ndarray:
        """Generate a single feature column.

        Parameters:
            dtype: data type of the feature, either `int` or `float`

        Return:
            feat: generated feature column
        """
        if dtype == "int":
            feat_col = np.random.randint(0, int(1e6), size=(self.n_samples, 1), dtype=np.int32)
        elif dtype == "float":
            feat_col = np.random.rand(self.n_samples, 1).astype(np.float32)
        feat_col = np.squeeze(feat_col, axis=1)

        return feat_col
