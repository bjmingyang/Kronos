import gc
import os
import sys
import re
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pytz
from matplotlib.dates import DateFormatter

from model import KronosTokenizer, Kronos, KronosPredictor

# --- Configuration ---
Config = {
    "REPO_PATH": Path(__file__).parent.resolve(),
    "MODEL_PATH": "../cache",
    "CSV_PATH": "~/.quant1x/5min/sz000/sz000776.csv",
    "SYMBOL": '000776',  # Extracted from CSV filename for display purposes
    "INTERVAL": '5min',  # Data interval (5 minutes for A-share example)
    "HIST_POINTS": 600,  # Lookback historical points (adapted from second code)
    "PRED_HORIZON": 60,  # Prediction horizon (adapted from second code)
    "N_PREDICTIONS": 120,  # Number of probabilistic samples (from first code, can set to 1 for deterministic)
    "VOL_WINDOW": 24,  # Window for volatility calculation (from first code)
}

def load_model():
    """从不同的本地路径加载微调后的 Kronos 模型和分词器。"""
    print("Loading fine-tuned Kronos model and tokenizer from separate local paths...")

    # 1. 定义分词器 (Tokenizer) 的路径
    finetuned_tokenizer_path = "/root/wangmy/Kronos/data/outputs/models/finetune_tokenizer_demo/checkpoints/best_model"

    # 2. 定义模型 (Model) 的路径
    finetuned_model_path = "/root/wangmy/Kronos/data/outputs/models/finetune_predictor_demo/checkpoints/best_model"

    # 3. 分别从对应的路径加载
    print(f"Loading tokenizer from: {finetuned_tokenizer_path}")
    tokenizer = KronosTokenizer.from_pretrained(finetuned_tokenizer_path)

    print(f"Loading model from: {finetuned_model_path}")
    model = Kronos.from_pretrained(finetuned_model_path)

    tokenizer.eval()
    model.eval()

    # 检查是否有可用的CUDA设备，否则使用CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print("Fine-tuned model and tokenizer loaded successfully.")

    return predictor

#def load_model():
#    """Loads the Kronos model and tokenizer (adapted from second code)."""
#    print("Loading Kronos model...")
#    finetuned_model_path = "/root/wangmy/Kronos/data/outputs/models/finetune_predictor_demo/checkpoints/best_model"
#    model = Kronos.from_pretrained(finetuned_model_path)
#    tokenizer = Kronos.from_pretrained
#    #tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", cache_dir=Config["MODEL_PATH"])
#    #model = Kronos.from_pretrained("NeoQuasar/Kronos-base", cache_dir=Config["MODEL_PATH"])
#    tokenizer.eval()
#    model.eval()
#    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
#    print("Model loaded successfully.")
#    return predictor
#

def make_prediction(df, predictor):
    """Generates probabilistic forecasts using the Kronos model (combines methods from both codes)."""
    last_timestamp = df['timestamps'].max()
    freq = '5min' if Config["INTERVAL"] == '5min' else 'H'  # Adapt frequency based on interval
    start_new_range = last_timestamp + pd.Timedelta(minutes=5) if freq == '5min' else last_timestamp + pd.Timedelta(hours=1)
    new_timestamps_index = pd.date_range(
        start=start_new_range,
        periods=Config["PRED_HORIZON"],
        freq=freq
    )
    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df['timestamps']
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    with torch.no_grad():
        print("Making main prediction (T=1.0)...")
        begin_time = time.time()
        close_preds_main, volume_preds_main = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=Config["PRED_HORIZON"], T=0.8, top_p=0.9,  # top_p from second code
            sample_count=Config["N_PREDICTIONS"], verbose=True
        )
        print(f"Main prediction completed in {time.time() - begin_time:.2f} seconds.")

        close_preds_volatility = close_preds_main  # Placeholder, as in first code's commented section

    return close_preds_main, volume_preds_main, close_preds_volatility


def fetch_data():
    """Loads data from a CSV file instead of fetching from Binance (adapted for A-share)."""
    print(f"Loading {Config['HIST_POINTS'] + Config['VOL_WINDOW']} points from {Config['CSV_PATH']}...")
    df = pd.read_csv(Config["CSV_PATH"])
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df['timestamps'] = df['timestamps'].dt.tz_localize('Asia/Shanghai')
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
    print("Data loaded successfully.")
    return df.tail(Config["HIST_POINTS"] + Config["VOL_WINDOW"])  # Limit to required history


def calculate_metrics(hist_df, close_preds_df, v_close_preds_df):
    """
    Calculates upside and volatility amplification probabilities (from first code, adapted for any interval).
    """
    last_close = hist_df['close'].iloc[-1]

    # Upside Probability
    final_hour_preds = close_preds_df.iloc[-1]
    upside_prob = (final_hour_preds > last_close).mean()

    # Volatility Amplification Probability
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-Config["VOL_WINDOW"]:].std()

    amplification_count = 0
    for col in v_close_preds_df.columns:
        full_sequence = pd.concat([pd.Series([last_close]), v_close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
        predicted_vol = pred_log_returns.std()
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(v_close_preds_df.columns)

    print(f"Upside Probability ({Config['PRED_HORIZON']} steps): {upside_prob:.2%}, Volatility Amplification Probability: {vol_amp_prob:.2%}")
    return upside_prob, vol_amp_prob


def create_plot(hist_df, close_preds_df, volume_preds_df):
    """Generates and saves a comprehensive forecast chart (from first code, adapted for probabilistic and A-share)."""
    print("Generating comprehensive forecast chart...")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Filter historical data to only today's data
    last_timestamp = hist_df['timestamps'].max()
    today_start = last_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    hist_df_today = hist_df[hist_df['timestamps'] >= today_start]

    hist_time = hist_df_today['timestamps']
    last_hist_time = hist_time.iloc[-1] if not hist_time.empty else last_timestamp
    delta = timedelta(minutes=5) if Config["INTERVAL"] == '5min' else timedelta(hours=1)
    pred_time = pd.to_datetime([last_hist_time + delta * (i + 1) for i in range(len(close_preds_df))])

    # Price plot (probabilistic, like first code)
    ax1.plot(hist_time, hist_df_today['close'], color='royalblue', label='Historical Price (Today)', linewidth=1.5)
    mean_preds = close_preds_df.mean(axis=1)
    ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
    ax1.fill_between(pred_time, close_preds_df.min(axis=1), close_preds_df.max(axis=1), color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
    ax1.set_title(f'{Config["SYMBOL"]} Probabilistic Price & Volume Forecast (Today + Next {Config["PRED_HORIZON"]} Steps)', fontsize=16, weight='bold')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Volume plot (bar for history, mean for forecast, adapted)
    width = 0.001 if Config["INTERVAL"] == '5min' else 0.03  # Adjust bar width for finer interval
    ax2.bar(hist_time, hist_df_today['volume'], color='skyblue', label='Historical Volume (Today)', width=width)
    ax2.bar(pred_time, volume_preds_df.mean(axis=1), color='sandybrown', label='Mean Forecasted Volume', width=width)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    separator_time = last_hist_time + (delta / 2)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5, label='_nolegend_')
        ax.tick_params(axis='x', rotation=30)

    # Set date formatter with Beijing timezone
    tz = pytz.timezone('Asia/Shanghai')
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M', tz=tz))
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M', tz=tz))

    fig.tight_layout()
    chart_path = Config["REPO_PATH"] / 'prediction_chart.png'
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")


def update_html(upside_prob, vol_amp_prob):
    """Updates the index.html file (from first code, optional if you have an index.html)."""
    print("Updating index.html...")
    html_path = Config["REPO_PATH"] / 'index.html'
    if not html_path.exists():
        print("index.html not found, skipping HTML update.")
        return
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    upside_prob_str = f'{upside_prob:.1%}'
    vol_amp_prob_str = f'{vol_amp_prob:.1%}'

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(
        r'(<strong id="update-time">).*?(</strong>)',
        lambda m: f'{m.group(1)}{now_str}{m.group(2)}',
        content
    )
    content = re.sub(
        r'(<p class="metric-value" id="upside-prob">).*?(</p>)',
        lambda m: f'{m.group(1)}{upside_prob_str}{m.group(2)}',
        content
    )
    content = re.sub(
        r'(<p class="metric-value" id="vol-amp-prob">).*?(</p>)',
        lambda m: f'{m.group(1)}{vol_amp_prob_str}{m.group(2)}',
        content
    )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("HTML file updated successfully.")


def git_commit_and_push(commit_message):
    """Adds, commits, and pushes files to Git (from first code, optional)."""
    print("Performing Git operations...")
    try:
        os.chdir(Config["REPO_PATH"])
        subprocess.run(['git', 'add', 'prediction_chart.png', 'index.html'], check=True, capture_output=True, text=True)
        commit_result = subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True, text=True)
        print(commit_result.stdout)
        push_result = subprocess.run(['git', 'push'], check=True, capture_output=True, text=True)
        print(push_result.stdout)
        print("Git push successful.")
    except subprocess.CalledProcessError as e:
        output = e.stdout if e.stdout else e.stderr
        if "nothing to commit" in output or "Your branch is up to date" in output:
            print("No new changes to commit or push.")
        else:
            print(f"A Git error occurred:\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}")
    except FileNotFoundError:
        print("Git not found or not a Git repository, skipping.")


def main_task(model):
    """Executes one full inference cycle (combines both codes' logic for file-based A-share prediction)."""
    print("\n" + "=" * 60 + f"\nStarting inference task at {datetime.now(timezone.utc)}\n" + "=" * 60)
    df = fetch_data()

    close_preds, volume_preds, v_close_preds = make_prediction(df, model)

    # For plot, use full df.tail(HIST_POINTS) but filter to today inside create_plot
    hist_df_for_plot = df.tail(Config["HIST_POINTS"])
    hist_df_for_metrics = df.tail(Config["VOL_WINDOW"])

    upside_prob, vol_amp_prob = calculate_metrics(hist_df_for_metrics, close_preds, v_close_preds)
    create_plot(hist_df_for_plot, close_preds, volume_preds)
    update_html(upside_prob, vol_amp_prob)

    commit_message = f"Auto-update forecast for {Config['SYMBOL']} at {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    git_commit_and_push(commit_message)

    # Memory cleanup (from first code)
    del df, close_preds, volume_preds, v_close_preds
    del hist_df_for_plot, hist_df_for_metrics
    gc.collect()

    print("-" * 60 + "\n--- Task completed successfully ---\n" + "-" * 60 + "\n")


if __name__ == '__main__':
    model_path = Path(Config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)

    loaded_model = load_model()
    main_task(loaded_model)  # Run once
    # If you want scheduling, uncomment the following:
    # run_scheduler(loaded_model)
