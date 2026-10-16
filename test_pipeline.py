import pandas as pd
from src.feature_extractor import FeatureExtractor
from src.model import JraModel
from src.backtest import JraBacktester

def run_pipeline_test():
    print("--- 1. データ読み込み ---")
    try:
        df = pd.read_csv('mock_race_data.csv')
    except FileNotFoundError:
        print("mock_race_data.csv が見つかりません。create_mock.py を先に実行してください。")
        return

    print("--- 2. 特徴量抽出 ---")
    extractor = FeatureExtractor(df)
    feature_df = extractor.generate_features()

    print("--- 3. 機械学習モデルの学習と予測 ---")
    model = JraModel()
    model.train(feature_df)
    pred_df = model.predict(feature_df)

    print("--- 4. バックテスト（詳細シミュレーション） ---")
    backtester = JraBacktester(ev_threshold=1.0, max_bets_per_race=2, bet_amount_per_horse=100)
    summary = backtester.run(pred_df)
    
    print(f" - 買い目総数（購入件数）: {summary['total_bets']} 件")
    print(f" - 的中数: {summary['total_hits']} 件 (的中率: {summary['hit_rate']:.2f} %)")
    print(f" - 総投資額: {summary['total_investment']} 円")
    print(f" - 総払戻額: {summary['total_payout']} 円")
    print(f" - 回収率 (ROI): {summary['roi']:.2f} %")
    
    print("\n--- 評価詳細データ ---")
    print(summary['details'][['horse_name', 'odds', 'pred_win_prob', 'expected_value', 'buy_flag', 'target']])

if __name__ == "__main__":
    run_pipeline_test()