import pandas as pd

class JraBacktester:
    """
    AIの予測勝率と実際のオッズから期待値を計算し、回収率をシミュレーションするクラス。
    """
    def __init__(self, threshold_ev: float = 1.0):
        """
        :param threshold_ev: 買うかどうかの基準となる期待値の閾値（例: 1.0以上なら購入）
        """
        self.threshold_ev = threshold_ev

    def calculate_expected_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        期待値 (EV) = 予測勝率 (pred_win_prob) × 単勝オッズ (odds) を計算する
        """
        df = df.copy()
        if 'pred_win_prob' not in df.columns or 'odds' not in df.columns:
            raise ValueError("データに 'pred_win_prob' または 'odds' 列が含まれていません。")

        # 期待値の計算
        df['expected_value'] = df['pred_win_prob'] * df['odds']
        
        # 期待値が閾値以上の場合に「購入フラグ」を立てる
        df['buy_flag'] = df['expected_value'] >= self.threshold_ev
        
        return df

    def run_backtest(self, df: pd.DataFrame) -> dict:
        """
        バックテストを実行し、総投資額、回収額、回収率（ROI）を計算する
        """
        df_ev = self.calculate_expected_value(df)
        
        # 購入対象とした馬を抽出
        bought_df = df_ev[df_ev['buy_flag'] == True]
        
        total_bets = len(bought_df)
        if total_bets == 0:
            return {
                "total_bets": 0,
                "total_investment": 0,
                "total_return": 0,
                "roi": 0.0,
                "evaluated_df": df_ev
            }

        # 1点あたりの賭け金を 100円 と仮定
        bet_amount = 100
        total_investment = total_bets * bet_amount
        
        # 的中した場合の払戻金計算（finish_order == 1 が的中）
        # ※簡易的に単勝馬券の回収をシミュレーション
        total_return = 0
        for _, row in bought_df.iterrows():
            if row.get('finish_order') == 1:
                total_return += bet_amount * row['odds']

        # 回収率 (ROI: Return on Investment) の計算（%表示）
        roi = (total_return / total_investment) * 100 if total_investment > 0 else 0.0

        results = {
            "total_bets": total_bets,
            "total_investment": total_investment,
            "total_return": total_return,
            "roi": roi,
            "evaluated_df": df_ev
        }
        return results