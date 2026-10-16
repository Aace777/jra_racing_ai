import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import optuna

# Optunaのログがうるさくならないように調整
optuna.logging.set_verbosity(optuna.logging.WARNING)

class JraModel:
    def __init__(self, params=None):
        """
        LightGBM モデルクラス（クロスバリデーション & Optuna対応）
        """
        self.models = []
        self.params = params if params is not None else {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 15,
            'max_depth': 4,
            'feature_fraction': 0.8,
            'seed': 42,
            'verbose': -1
        }

    def optimize_hyperparameters(self, df: pd.DataFrame, feature_cols: list, n_trials=20):
        """
        Optunaを用いて過学習を防ぎながら最適なハイパーパラメータを探索する
        """
        print(f"🔬 Optunaによるハイパーパラメータ自動最適化を開始します（試行回数: {n_trials}回）...")
        
        X = df[feature_cols]
        # 的中（1着）を予測するため、finish_order == 1 または target をターゲットにする
        if 'target' in df.columns:
            y = df['target']
        elif 'finish_order' in df.columns:
            y = (df['finish_order'] == 1).astype(int)
        else:
            raise ValueError("目的変数となる列（target または finish_order）が見つかりません。")
            
        groups = df['race_id'] if 'race_id' in df.columns else np.arange(len(df))

        def objective(trial):
            # 探索するパラメータ空間
            param = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 8, 31),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 20),
                'seed': 42,
                'verbose': -1
            }

            # レース単位のグループクロスバリデーション（GroupKFold）で汎用性を評価
            gkf = GroupKFold(n_splits=3)
            scores = []

            for train_idx, val_idx in gkf.split(X, y, groups):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

                model = lgb.train(
                    param,
                    train_data,
                    num_boost_round=200,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                )

                preds = model.predict(X_val)
                # 簡易的な評価指標としてログロスを使用（低いほど良い）
                score = np.mean((preds - y_val) ** 2)
                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        print("✨ 最適化が完了しました！最適パラメータ:")
        print(study.best_params)

        # 見つかった最適パラメータをベースにモデルの設定をアップデート
        self.params.update(study.best_params)

    def train(self, df: pd.DataFrame, feature_cols: list = None):
        """
        GroupKFold クロスバリデーションを用いて複数モデルを学習する（過学習防止）
        """
        print("🚀 LightGBMモデルのクロスバリデーション学習を開始します...")
        
        if feature_cols is None:
            # デフォルトの特徴量リスト
            feature_cols = [
                'odds', 'weight', 'past_avg_rank', 'days_since_last_race',
                'odds_zscore', 'weight_diff_from_avg', 'past_rank_in_race',
                'is_nige', 'race_nige_count'
            ]

        X = df[feature_cols]
        if 'target' in df.columns:
            y = df['target']
        elif 'finish_order' in df.columns:
            y = (df['finish_order'] == 1).astype(int)
        else:
            raise ValueError("目的変数が見つかりません。")

        groups = df['race_id'] if 'race_id' in df.columns else np.arange(len(df))

        gkf = GroupKFold(n_splits=3)
        self.models = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            self.models.append(model)

        print(f"✅ {len(self.models)}個のモデルのクロスバリデーション学習が完了しました！")

    def predict(self, df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
        """
        学習済み複数モデルの予測確率を平均化して出力する（アンサンブル効果で精度向上）
        """
        if feature_cols is None:
            feature_cols = [
                'odds', 'weight', 'past_avg_rank', 'days_since_last_race',
                'odds_zscore', 'weight_diff_from_avg', 'past_rank_in_race',
                'is_nige', 'race_nige_count'
            ]

        X = df[feature_cols]
        
        # すべてのFoldモデルの予測を平均する
        preds_list = [model.predict(X) for model in self.models]
        avg_preds = np.mean(preds_list, axis=0)

        result_df = df.copy()
        result_df['pred_win_prob'] = avg_preds
        return result_df