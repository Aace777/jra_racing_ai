import pandas as pd
import numpy as np

class FeatureExtractor:
    def __init__(self, df=None):
        self.df = df

    def extract_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        モックデータまたは本番データを入力として受け取り、
        LightGBM用の高度な特徴量を追加したDataFrameを返す。
        """
        if df is None:
            df = self.df
        
        if df is None:
            raise ValueError("DataFrameが渡されていません。")

        # 元データを汚さないようにコピー
        features = df.copy()

        # 1. 基本的な数値化・欠損値処理（カラムが存在する場合のみ安全に処理）
        if 'odds' in features.columns:
            features['odds'] = pd.to_numeric(features['odds'], errors='coerce').fillna(1.0)
        else:
            features['odds'] = 1.0

        if 'weight' in features.columns:
            features['weight'] = pd.to_numeric(features['weight'], errors='coerce').fillna(450.0)
        else:
            features['weight'] = 450.0

        # 2. 過去走集計の特徴量（モック用シミュレーション）
        if 'past_avg_rank' not in features.columns:
            features['past_avg_rank'] = np.random.uniform(1.0, 15.0, len(features))
            
        if 'days_since_last_race' not in features.columns:
            features['days_since_last_race'] = np.random.randint(14, 180, len(features))

        # 3. レース内相対化の特徴量（Zスコア・順位化）
        if 'race_id' in features.columns:
            features['odds_zscore'] = features.groupby('race_id')['odds'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-5)
            )
            features['weight_diff_from_avg'] = features.groupby('race_id')['weight'].transform(
                lambda x: x - x.mean()
            )
            features['past_rank_in_race'] = features.groupby('race_id')['past_avg_rank'].transform(
                lambda x: x.rank(ascending=True)
            )
        else:
            features['odds_zscore'] = 0.0
            features['weight_diff_from_avg'] = 0.0
            features['past_rank_in_race'] = 1.0

        # 4. カテゴリカル変数のエンコーディング準備
        if 'jockey_id' in features.columns:
            features['jockey_id'] = features['jockey_id'].astype('category')
        if 'trainer_id' in features.columns:
            features['trainer_id'] = features['trainer_id'].astype('category')

        return features

    def generate_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """extract_featuresへのエイリアス（互換用）"""
        return self.extract_features(df)
    def get_feature_columns(self) -> list:
        """
        LightGBMの学習に使用する特徴量カラム名のリストを返す
        """
        return [
            'odds',
            'weight',
            'past_avg_rank',
            'days_since_last_race',
            'odds_zscore',
            'weight_diff_from_avg',
            'past_rank_in_race'
        ]