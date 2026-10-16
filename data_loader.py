import pandas as pd
from pathlib import Path

class JraVanDataLoader:
    """
    JRA-VANから出力されるデータを安全に読み込み、
    標準的なPandas DataFrameに変換する専用インポーター。
    """
    def __init__(self, data_dir: str = "data/raw/"):
        self.data_dir = Path(data_dir)

    def load_race_data(self, filename: str = "race_results.csv") -> pd.DataFrame:
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")

        # 文字コード補正処理
        df = self._read_with_fallback(file_path)
        
        # 基本クリーニング
        df = self._clean_data(df)
        return df

    def _read_with_fallback(self, file_path: Path) -> pd.DataFrame:
        encodings = ["utf-8-sig", "cp932", "shift_jis"]
        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"文字コードを特定できませんでした: {file_path}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df