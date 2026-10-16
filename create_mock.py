import pandas as pd
import numpy as np

def create_mock_data():
    np.random.seed(42)
    n_samples = 40
    
    # 4レース分のモックデータ（1レースあたり10頭立て）
    race_ids = [f"2026{i:02d}" for i in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
    
    styles = ['逃げ', '先行', '差し', '追込']
    
    df = pd.DataFrame({
        'race_id': race_ids,
        'horse_name': [f"馬_{i}" for i in range(n_samples)],
        'odds': np.random.uniform(1.5, 30.0, n_samples),
        'weight': np.random.uniform(400, 520, n_samples),
        'running_style': np.random.choice(styles, n_samples),
        'finish_order': np.random.randint(1, 11, n_samples), # 着順（1〜10位）を追加
        'target': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]) # 1着フラグ
    })
    
    df.to_csv('mock_race_data.csv', index=False)
    print("モックデータを更新しました（finish_order追加）！")

if __name__ == "__main__":
    create_mock_data()