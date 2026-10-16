import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

class JraModel:
    """
    LightGBMを使用してJRAのレース結果（勝敗や着順）を予測するモデルクラス。
    """
    def __init__(self):
        self.model = None
        # 予測モデルの訓練に使う特徴量（カラム名）のリスト
        self.feature_cols = [
            'horse_weight', 
            'weight_change', 
            'weight_large_change_flag', 
            'heavy_weight_flag', 
            'market_implied_prob'
        ]

    def train(self, df: pd.DataFrame):
        """
        特徴量データフレームを受け取り、モデルを学習させる
        """
        # 目的変数（正解）の作成：今回は「1着になったかどうか（1 or 0）」を予測のターゲットにする例
        df = df.copy()
        if 'finish_order' in df.columns:
            df['target'] = df['finish_order'].apply(lambda x: 1 if x == 1 else 0)
        else:
            raise ValueError("データに 'finish_order' 列が含まれていません。")

        # 欠損値の除外など
        train_df = df.dropna(subset=self.feature_cols + ['target'])

        X = train_df[self.feature_cols]
        y = train_df['target']

        # 訓練データと検証データに分割（モックが少ない場合はそのまま学習することも考慮）
        if len(train_df) >= 4:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
        else:
            # データが少なすぎるモックテスト用のフォールバック
            X_train, X_val, y_train, y_val = X, X, y, y

        # LightGBMのデータセット作成
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        # ハイパーパラメータの設定
        # ハイパーパラメータの設定
        params = {
            'objective': 'binary',      # 二値分類（勝つか負けるか）
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbose': -1
        }

        print("🚀 LightGBMモデルの学習を開始します...")
        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_eval],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        print("✅ モデルの学習が完了しました！")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        未知のデータ（または予測対象データ）を入力し、勝率の予測確率を出力する
        """
        if self.model is None:
            raise ValueError("モデルがまだ学習されていません。先に train() を実行してください。")
        
        X = df[self.feature_cols]
        preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        result_df = df.copy()
        result_df['pred_win_prob'] = preds
        return result_df