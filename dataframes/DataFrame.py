import numpy as np


class DataFrame(np.ndarray):
    "The Object Dataframe we're using for data."

    def __new__(cls, input_array, dtypes=None):
        obj = np.asanyarray(input_array).view(cls)
        obj.dtypes = dtypes or {}
        return obj

    def __array_finalize__(self, obj):
        self.dtypes = getattr(obj, "dtypes", {})

    @staticmethod
    def _try_cast(column):
        try:
            return column.astype(np.float64)
        except (ValueError, TypeError):
            return column

    @staticmethod
    def _infer_dtype(casted):
        if casted.dtype == float:
            non_nan = casted[~np.isnan(casted)]
            if non_nan.size == 0:
                return "float"
            if all(v in (0.0, 1.0) for v in non_nan):
                return "bool"
            if all(v.is_integer() for v in non_nan):
                return "int"
            return "float"
        return "str"

    @staticmethod
    def load_csv(source):
        import csv
        from io import StringIO

        # Read raw text
        if hasattr(source, 'read'):
            raw = source.read()
        else:
            with open(source, 'r', encoding='utf-8') as f:
                raw = f.read()

        reader = csv.reader(StringIO(raw))
        rows = list(reader)

        if len(rows) < 2:
            raise ValueError("CSV must have a header and at least one data row.")

        headers = [h.strip().strip('"') for h in rows[0]]

        def clean(val):
            val = val.strip().strip('"')
            if val in ('', '?', 'NA', 'N/A', 'nan', 'NaN'):
                return np.nan
            return val

        # Build column arrays
        n_rows = len(rows) - 1
        n_cols = len(headers)
        raw_cols = {h: [] for h in headers}

        for row in rows[1:]:
            # pad or trim row to match header length
            while len(row) < n_cols:
                row.append('')
            for i, h in enumerate(headers):
                raw_cols[h].append(clean(row[i]))

        dtypes = {}
        cleaned = {}

        for name in headers:
            col = np.array(raw_cols[name], dtype=object)
            casted = DataFrame._try_cast(col)
            cleaned[name] = casted
            dtypes[name] = DataFrame._infer_dtype(casted)

        # Build structured array
        dt = np.dtype([(name, object) for name in headers])
        data = np.empty(n_rows, dtype=dt)
        for name in headers:
            data[name] = cleaned[name]

        return DataFrame(data, dtypes=dtypes)

    def get_numerical(self):
        columns = [
            name for name, value in self.dtypes.items() if value in ("bool", "int", "float")
        ]
        view = self[columns]
        view.dtypes = {key: self.dtypes[key] for key in columns}
        return view

    def get_object(self):
        columns = [name for name, value in self.dtypes.items() if value == "str"]
        view = self[columns]
        view.dtypes = {key: self.dtypes[key] for key in columns}

        return view

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, str) and self.dtypes.get(key) in ("bool", "int", "float"):
            return result.astype(np.float64)
        return result

    def format(self):
        return self.shape[0], len(self.dtypes.keys())

    def get_features(self):
        return list(self.dtypes.keys())

    def get_type(self, column_name: str):
        return self.dtypes[column_name]

    def count_values(self):
        return dict(zip(*np.unique(self, return_counts=True)))


if __name__ == "__main__":
    df = DataFrame.load_csv("data.csv")
    print(df.get_numerical().get_features())
    print(df.get_object().get_features())
    print(df.get_numerical().get_type("age"))
    print(df["age"].count_values())
