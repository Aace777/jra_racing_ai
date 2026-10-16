from src.data_loader import JraVanDataLoader
from src.feature_extractor import FeatureExtractor
from src.model import JraModel
from src.backtest import JraBacktester

def run_pipeline_test():
    print("--- 1. データ読み込み ---")
    loader = JraVanDataLoader()
    raw_df = loader.load_race_data("race_results.csv")

    print("\n--- 2. 特徴量抽出 ---")
    extractor = FeatureExtractor(raw_df)
    feature_df = extractor.generate_features()

    print("\n--- 3. 機械学習モデルの学習と予測 ---")
    model = JraModel()
    model.train(feature_df)
    pred_df = model.predict(feature_df)

    print("\n--- 4. バックテスト（回収率シミュレーション） ---")
    # 期待値1.0以上の馬券を狙う設定
    backtester = JraBacktester(threshold_ev=1.0)
    result = backtester.run_backtest(pred_df)

    print(f"  - 買い目総数: {result['total_bets']} 件")
    print(f"  - 総投資額: {result['total_investment']} 円")
    print(f"  - 総払戻額: {result['total_return']} 円")
    print(f"  - 回収率 (ROI): {result['roi']:.2f} %")

    print("\n--- 評価詳細データ ---")
    print(result['evaluated_df'][['horse_name', 'odds', 'pred_win_prob', 'expected_value', 'buy_flag', 'finish_order']])

if __name__ == "__main__":
    run_pipeline_test()