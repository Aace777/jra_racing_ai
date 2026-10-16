import pandas as pd

class FeatureExtractor:
    """
    JRA-VAN公式データ（またはモック）から、AI予測用の強力な特徴量を生成するクラス。
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def generate_features(self) -> pd.DataFrame:
        """すべての特徴量生成プロセスを順次実行するメインパイプライン"""
        df = self.df
        
        # 1. 馬体重に関する特徴量
        df = self._extract_weight_features(df)
        
        # 2. オッズから逆算した市場評価（確率）の特徴量
        df = self._extract_market_features(df)
        
        # 3. 今後追加する高度な指標のプレースホルダー（例: タイム差やローテーション）
        df = self._extract_stub_features(df)
        
        return df

    def _extract_weight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬体重の増減に関するフラグやトレンドを作成"""
        if 'weight_change' in df.columns:
            # 大幅な馬体重増減（±10kg以上）は激走や不調のサインになりやすい
            df['weight_large_change_flag'] = df['weight_change'].apply(lambda x: 1 if abs(x) >= 10 else 0)
            
            # 馬体重そのものの絶対値（重馬場適性などのベースになる）
            if 'horse_weight' in df.columns:
                df['heavy_weight_flag'] = df['horse_weight'].apply(lambda x: 1 if x >= 500 else 0)
                
        return df

    def _extract_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """オッズから市場の期待値を逆算する特徴量"""
        if 'odds' in df.columns:
            # オッズの逆数 = 簡易的な市場支持率（確率のベース）
            df['market_implied_prob'] = 1 / df['odds']
            
            # 人気の極端さを表す対数変換などもここで可能
        return df

    def _extract_stub_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [将来拡張用] 
        ここに「過去N走の平均着順」や「コース・距離別勝率」などを計算するロジックを追加していきます。
        """
        # 現段階ではダミーとしてプレースホルダーを設置
        df['is_ready_for_ml'] = 1
        return df