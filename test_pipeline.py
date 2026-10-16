from src.data_loader import JraVanDataLoader
from src.feature_extractor import FeatureExtractor

def run_pipeline_test():
    print("--- 1. データ読み込みテスト ---")
    loader = JraVanDataLoader()
    raw_df = loader.load_race_data("race_results.csv")
    print(f"読み込み完了: {raw_df.shape[0]} 行 / {raw_df.shape[1]} 列")
    print(raw_df.head(2))

    print("\n--- 2. 特徴量抽出テスト ---")
    extractor = FeatureExtractor(raw_df)
    feature_df = extractor.generate_features()
    print(f"特徴量生成完了: {feature_df.shape[0]} 行 / {feature_df.shape[1]} 列")
    
    # 修正：新しく追加された特徴量の列名を指定
    print(feature_df[['horse_name', 'weight_change', 'weight_large_change_flag', 'heavy_weight_flag', 'market_implied_prob']])

if __name__ == "__main__":
    run_pipeline_test()