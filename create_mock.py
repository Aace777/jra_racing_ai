import pandas as pd
from pathlib import Path

def generate_mock_csv():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    mock_data = {
        "race_id": ["202610010101", "202610010101", "202610010101", "202610010102"],
        "date": ["2026-06-01", "2026-06-01", "2026-06-01", "2026-06-01"],
        "horse_name": ["テイエムサンプル", "ダイワテスト", "オルフェコード", "ディープモック"],
        "horse_weight": [510, 472, 490, 460],     # 510kgの馬を追加（重馬場フラグテスト用）
        "weight_change": [12, -4, 2, -10],         # ±10kg以上の変化を追加
        "odds": [2.5, 5.1, 12.4, 1.8],
        "finish_order": [1, 2, 3, 1]
    }

    df = pd.DataFrame(mock_data)
    target_path = raw_dir / "race_results.csv"
    df.to_csv(target_path, index=False, encoding="utf-8-sig")
    print(f"✅ リッチ化されたモックデータを作成しました: {target_path}")

if __name__ == "__main__":
    generate_mock_csv()