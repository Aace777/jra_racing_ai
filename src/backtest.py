import pandas as pd
import numpy as np

class BacktestSimulator:
    def __init__(self, ev_threshold=1.0, max_bets_per_race=2, bet_amount_per_horse=100):
        """
        高度な期待値フィルター＆資金配分バックテストシミュレーター
        - ev_threshold: 購入する期待値の最低ライン
        - max_bets_per_race: 1レースあたりの最大購入点数（トリガミ防止）
        - bet_amount_per_horse: 1頭あたりの基本投資額（円）
        """
        self.ev_threshold = ev_threshold
        self.max_bets_per_race = max_bets_per_race
        self.bet_amount_per_horse = bet_amount_per_horse

    def run(self, df: pd.DataFrame) -> dict:
        """
        予測確率とオッズから期待値を計算し、高度な絞り込みを行ってバックテストを実行する
        """
        results = df.copy()

        # 1. 期待値（EV = 予測確率 × オッズ）の計算
        if 'pred_win_prob' not in results.columns:
            raise ValueError("予測確率カラム 'pred_win_prob' が存在しません。")
        if 'odds' not in results.columns:
            raise ValueError("オッズカラム 'odds' が存在しません。")

        results['expected_value'] = results['pred_win_prob'] * results['odds']

        # 2. 購入フラグの初期化
        results['buy_flag'] = False

        # 3. レースごとの購入点数制限・絞り込み処理（期待値上位N頭のみ購入）
        for race_id, group in results.groupby('race_id'):
            eligible = group[group['expected_value'] >= self.ev_threshold]
            if not eligible.empty:
                top_targets = eligible.sort_values(by='expected_value', ascending=False).head(self.max_bets_per_race)
                results.loc[top_targets.index, 'buy_flag'] = True

        # 4. 収支・払戻金の計算
        if 'target' not in results.columns:
            results['target'] = 0

        results['investment'] = 0
        results['payout'] = 0

        bought_mask = results['buy_flag'] == True
        results.loc[bought_mask, 'investment'] = self.bet_amount_per_horse

        hit_and_bought = (results['buy_flag'] == True) & (results['target'] == 1)
        results.loc[hit_and_bought, 'payout'] = results.loc[hit_and_bought, 'investment'] * results.loc[hit_and_bought, 'odds']

        # 5. 全体サマリーの集計
        total_bets = bought_mask.sum()
        total_investment = results['investment'].sum()
        total_payout = results['payout'].sum()
        total_hits = hit_and_bought.sum()
        
        roi = (total_payout / total_investment * 100) if total_investment > 0 else 0.0
        hit_rate = (total_hits / total_bets * 100) if total_bets > 0 else 0.0

        summary = {
            'total_bets': int(total_bets),
            'total_hits': int(total_hits),
            'total_investment': int(total_investment),
            'total_payout': int(total_payout),
            'roi': float(roi),
            'hit_rate': float(hit_rate),
            'details': results
        }

        return summary

# 互換性のためのエイリアス定義（JraBacktesterという名前でも呼び出せるようにする）
JraBacktester = BacktestSimulator