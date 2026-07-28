"""CodeTable — string <-> int32 code assignment, vocabulary-sized not row-sized.

Shared by RouteLegsIndex (may/venue_distributor/route_legs_index.py) and
OriginDestinationMatrixSource (may/attribute_assignment/data_sources.py), both
of which replace a per-row Python dict/tuple index with columnar numpy arrays
to stay memory-bounded at multi-million-row scale.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class CodeTable:
    """String <-> int32 code, assigned in first-appearance order.

    Encoding is vectorised per chunk via pandas.factorize: the per-row cost
    is paid only once per *unique* value in the chunk, not once per row. The
    running `_code_of` dict is sized by vocabulary, never by row count.
    """

    def __init__(self):
        self._code_of: Dict[str, int] = {}
        self.decode: List[str] = []

    def encode_series(self, series: pd.Series) -> np.ndarray:
        local_codes, uniques = pd.factorize(series.to_numpy())
        if (local_codes == -1).any():
            raise ValueError("CodeTable.encode_series: missing/NaN value in series")
        global_of_local = np.empty(len(uniques), dtype=np.int32)
        code_of = self._code_of
        decode = self.decode
        for local_code, value in enumerate(uniques):
            code = code_of.get(value)
            if code is None:
                code = len(decode)
                code_of[value] = code
                decode.append(value)
            global_of_local[local_code] = code
        return global_of_local[local_codes]

    def code_of(self, value: str) -> Optional[int]:
        return self._code_of.get(value)

    def __getitem__(self, code: int) -> str:
        return self.decode[code]

    def __len__(self) -> int:
        return len(self.decode)
