# ===================================================================
# DeBERTa v3-large 衣服レビュー推薦予測モデル
# ===================================================================
# このコードは、Microsoft DeBERTa v3-largeモデルを使用して
# 衣服のレビューデータから「レビュワーがその商品を推薦するか」を
# 予測するテキスト分類モデルを構築します。
#
# 手法:
# 1. DeBERTa v3-large事前学習済みモデルのファインチューニング
# 2. 3フォールド層化交差検証による汎化性能の向上
# 3. 早期終了による過学習の防止
# 4. AUCスコアによる評価
# ===================================================================

# 必要なライブラリのインポート
import os
import gc
import warnings
warnings.filterwarnings('ignore')  # 警告メッセージを非表示
import random
import math
from pathlib import Path

import json
import argparse
from itertools import chain
from functools import partial

# PyTorchとTransformers関連
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from tokenizers import AddedToken
from datasets import Dataset, features

# データ処理・評価関連
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, log_loss, roc_auc_score, matthews_corrcoef, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold

# ===================================================================
# 出力ディレクトリの作成
# ===================================================================
!mkdir oof      # Out-of-fold予測結果保存用ディレクトリ
!mkdir models   # 学習済みモデル保存用ディレクトリ

# ===================================================================
# 設定パラメータ
# ===================================================================
class CFG:
    """モデル学習・予測に関する設定パラメータを管理するクラス"""
    
    VER = 1                                    # バージョン番号
    AUTHOR = 'noname'                          # 作成者名
    COMPETITION = 'atmacup17'                  # コンペティション名
    
    # データパス設定
    DATA_PATH = Path('/kaggle/input/atmacup17dataset/atmacup17_dataset')
    OOF_DATA_PATH = Path('./oof')              # Out-of-fold予測結果保存先
    MODEL_DATA_PATH = Path('./models')         # モデル保存先
    
    # モデル設定
    MODEL_PATH = "microsoft/deberta-v3-large"  # 使用する事前学習済みモデル
    MAX_LENGTH = 256                           # 入力テキストの最大トークン長
    
    # 学習設定
    STEPS = 25                                 # 評価・保存を行うステップ間隔
    USE_GPU = torch.cuda.is_available()        # GPU使用可否の自動判定
    SEED = 0                                   # 再現性確保のための乱数シード
    N_SPLIT = 3                                # 交差検証の分割数
    
    # タスク設定
    target_col = 'Recommended IND'             # 予測対象列名
    target_col_class_num = 2                   # クラス数（2値分類）
    metric = 'auc'                             # 評価指標
    metric_maximize_flag = True                # 指標の最大化フラグ

# ===================================================================
# 乱数シード固定関数
# ===================================================================
def seed_everything(seed):
    """
    再現性確保のため、全ての乱数シードを固定する関数
    
    Args:
        seed (int): 設定する乱数シード値
    """
    random.seed(seed)                          # Python標準のrandomモジュール
    np.random.seed(seed)                       # NumPy
    os.environ['PYTHONHASHSEED'] = str(seed)   # Pythonハッシュ関数
    torch.manual_seed(seed)                    # PyTorch CPU
    
    if CFG.USE_GPU:
        torch.cuda.manual_seed(seed)           # PyTorch GPU
        torch.backends.cudnn.deterministic = True   # cuDNNの決定的動作
        torch.backends.cudnn.benchmark = False      # cuDNNベンチマークを無効化

# シード値の適用
seed_everything(CFG.SEED)

# 使用デバイスの設定
device = torch.device('cuda' if CFG.USE_GPU else 'cpu')

# ===================================================================
# データ読み込み
# ===================================================================
# 衣服マスターデータの読み込み
clothing_master_df = pd.read_csv(CFG.DATA_PATH / 'clothing_master.csv')

# 学習・テストデータの読み込み
train_df = pd.read_csv(CFG.DATA_PATH / 'train.csv')
test_df = pd.read_csv(CFG.DATA_PATH / 'test.csv')

# デバッグ用: データの一部のみ使用する場合
# train_df = pd.read_csv(CFG.DATA_PATH / 'train.csv').head(100)

# ===================================================================
# データ前処理関数
# ===================================================================
def preprocessing(df, clothing_master_df):
    """
    テキストデータの前処理を行う関数
    
    Args:
        df (pd.DataFrame): 処理対象のデータフレーム
        clothing_master_df (pd.DataFrame): 衣服マスターデータ
    
    Returns:
        pd.DataFrame: 前処理済みデータフレーム
    """
    # 欠損値の処理
    df['Title'] = df['Title'].fillna('')           # タイトルの欠損値を空文字で埋める
    df['Review Text'] = df['Review Text'].fillna('')  # レビューテキストの欠損値を空文字で埋める
    
    # タイトルとレビューテキストを結合してプロンプトを作成
    df['prompt'] = df['Title'] + ' ' + df['Review Text']
    
    return df

# データ前処理の実行
train_df = preprocessing(train_df, clothing_master_df)
test_df = preprocessing(test_df, clothing_master_df)

# ターゲット変数の準備（学習データのみ）
train_df['labels'] = train_df[CFG.target_col].astype(np.int8)

# ===================================================================
# モデル設定・評価関数
# ===================================================================
# トークナイザーの初期化
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_PATH)

def tokenize(sample):
    """
    テキストをトークン化する関数
    
    Args:
        sample (dict): 'prompt'キーを含む辞書
    
    Returns:
        dict: トークン化された結果（input_ids, attention_mask等）
    """
    return tokenizer(
        sample['prompt'], 
        max_length=CFG.MAX_LENGTH, 
        truncation=True  # 最大長を超える部分は切り捨て
    )

def compute_metrics(p):
    """
    モデルの評価指標を計算する関数
    
    Args:
        p (tuple): (予測値, 正解ラベル)のタプル
    
    Returns:
        dict: 評価指標の辞書
    """
    preds, labels = p
    # 予測値をソフトマックスで確率に変換
    preds = torch.softmax(torch.tensor(preds), dim=1).numpy()
    # AUCスコアを計算（クラス1の確率を使用）
    score = roc_auc_score(labels, preds[:, 1])
    return {'auc': score}

# ===================================================================
# 交差検証による学習・予測
# ===================================================================
# Out-of-fold予測結果格納用配列
predictions = np.zeros((len(train_df), CFG.target_col_class_num))

# 層化K分割交差検証の設定
kfold = StratifiedKFold(n_splits=CFG.N_SPLIT, shuffle=True, random_state=CFG.SEED)

# 各フォールドでの学習実行
for fold, (train_index, valid_index) in enumerate(kfold.split(train_df, train_df['Rating'])):
    print(f"=== Fold {fold + 1}/{CFG.N_SPLIT} ===")
    
    # データ分割
    ds_train = Dataset.from_pandas(train_df.iloc[train_index][['prompt', 'labels']].copy())
    ds_eval = Dataset.from_pandas(train_df.iloc[valid_index][['prompt', 'labels']].copy())

    # トークン化処理と不要列の削除
    ds_train = ds_train.map(tokenize).remove_columns(['prompt', '__index_level_0__'])
    ds_eval = ds_eval.map(tokenize).remove_columns(['prompt', '__index_level_0__'])

    # ===================================================================
    # 学習設定
    # ===================================================================
    train_args = TrainingArguments(
        # 出力設定
        output_dir=CFG.MODEL_DATA_PATH / f'/deberta-large-fold{fold}',
        report_to="none",                      # WandB等の外部ログ無効化
        
        # 学習ハイパーパラメータ
        fp16=True,                             # 半精度浮動小数点演算（高速化・メモリ節約）
        learning_rate=1e-5,                    # 学習率
        num_train_epochs=2,                    # エポック数
        per_device_train_batch_size=2,         # 学習時バッチサイズ
        per_device_eval_batch_size=4,          # 評価時バッチサイズ
        gradient_accumulation_steps=4,         # 勾配累積ステップ数
        
        # 評価・保存設定
        evaluation_strategy="steps",           # ステップ単位で評価
        do_eval=True,                          # 評価を実行
        eval_steps=CFG.STEPS,                  # 評価実行間隔
        save_total_limit=3,                    # 保存するチェックポイント数の上限
        save_strategy="steps",                 # ステップ単位で保存
        save_steps=CFG.STEPS,                  # 保存実行間隔
        logging_steps=CFG.STEPS,               # ログ出力間隔
        
        # 最適化設定
        lr_scheduler_type='linear',            # 線形学習率スケジューラ
        warmup_ratio=0.1,                     # ウォームアップ期間の割合
        weight_decay=0.01,                     # 重み減衰（L2正則化）
        
        # モデル選択設定
        metric_for_best_model="auc",           # 最良モデル判定基準
        greater_is_better=True,                # 指標が高いほど良い
        load_best_model_at_end=True,           # 学習終了時に最良モデルをロード
        
        # 再現性設定
        save_safetensors=True,                 # SafeTensors形式で保存
        seed=CFG.SEED,                         # モデルシード
        data_seed=CFG.SEED,                    # データシード
    )

    # ===================================================================
    # モデル初期化
    # ===================================================================
    config = AutoConfig.from_pretrained(CFG.MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(CFG.MODEL_PATH, config=config)

    # ===================================================================
    # トレーナー初期化
    # ===================================================================
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=DataCollatorWithPadding(tokenizer),  # 動的パディング
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,       # 評価指標計算関数
        callbacks=[EarlyStoppingCallback(      # 早期終了設定
            early_stopping_patience=3,         # 改善しないエポック数の上限
            early_stopping_threshold=0.01      # 改善と判定する最小閾値
        )],
    )

    # ===================================================================
    # モデル学習実行
    # ===================================================================
    trainer.train()
    
    # 評価結果の表示
    eval_results = trainer.evaluate(ds_eval)
    print(f"Fold {fold} AUC: {eval_results['eval_auc']:.6f}")

    # ===================================================================
    # モデル・トークナイザーの保存
    # ===================================================================
    model_save_path = f"deberta-large-seed{CFG.SEED}-Ver{CFG.VER}/deberta-large-fold{fold}"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # ===================================================================
    # Out-of-fold予測の保存
    # ===================================================================
    fold_predictions = trainer.predict(ds_eval).predictions
    fold_probs = torch.softmax(torch.tensor(fold_predictions), dim=1).numpy()
    predictions[valid_index] = fold_probs

# ===================================================================
# 学習結果の保存
# ===================================================================
# 予測確率を学習データに追加
train_df[f'deberta_large_Ver{CFG.VER}_pred_prob'] = predictions[:, 1]

# 予測結果付き学習データの保存
train_df.to_csv(f'./models/deberta-large-seed{CFG.SEED}-Ver{CFG.VER}-train-predictions.csv', index=False)

# Out-of-fold結果の保存
train_df.to_csv(f'./oof/deberta-large-seed{CFG.SEED}-Ver{CFG.VER}.csv', index=False)

# ===================================================================
# テストデータの予測
# ===================================================================
print("=== テストデータの予測開始 ===")

# テストデータのトークン化
ds_test = Dataset.from_pandas(test_df[['prompt']].copy())
ds_test = ds_test.map(tokenize)

# テスト用の学習設定
test_args = TrainingArguments(
    output_dir=CFG.MODEL_DATA_PATH / f'/deberta-large-fold{fold}',
    report_to="none",
    per_device_eval_batch_size=4,
    do_eval=False,                             # 評価は実行しない
    logging_strategy="no",                     # ログ無効化
    save_strategy="no",                        # 保存無効化
)

# テスト予測結果格納用配列
test_predictions = np.zeros((len(test_df), CFG.target_col_class_num))

# 各フォールドのモデルで予測し、アンサンブル
for fold in range(CFG.N_SPLIT):
    print(f"Fold {fold + 1} でテストデータを予測中...")
    
    # 学習済みモデルとトークナイザーの読み込み
    model_path = f"deberta-large-seed{CFG.SEED}-Ver{CFG.VER}/deberta-large-fold{fold}"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 予測専用トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=test_args,
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
    )

    # テストデータの予測実行
    test_predictions += torch.softmax(
        torch.tensor(trainer.predict(ds_test).predictions), dim=1
    ).numpy()

# アンサンブル（平均化）
test_predictions /= CFG.N_SPLIT

# ===================================================================
# サブミッション作成
# ===================================================================
# サブミッション用データフレーム作成
submission_df = pd.DataFrame({
    "target": test_predictions[:, 1]  # クラス1（推薦）の確率
})

# サブミッションファイルの保存
submission_df.to_csv('submission016.csv', index=False)

# テストデータに予測結果を追加（後続処理用）
test_df[f'deberta_large_Ver{CFG.VER}_pred_prob'] = test_predictions[:, 1]
test_df.to_csv(f'./models/deberta-large-seed{CFG.SEED}-Ver{CFG.VER}-test-predictions.csv', index=False)

print("=== 予測完了 ===")
print(f"サブミッションファイル: submission016.csv")
print(f"予測結果サンプル:")
print(submission_df.head())

# ===================================================================
# 最終結果の表示
# ===================================================================
overall_auc = roc_auc_score(train_df['labels'], train_df[f'deberta_large_Ver{CFG.VER}_pred_prob'])
print(f"Overall Cross-Validation AUC: {overall_auc:.6f}")

# メモリ解放
gc.collect()
torch.cuda.empty_cache() if CFG.USE_GPU else None
