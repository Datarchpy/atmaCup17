# ===================================================================
# 衣服レビュー推薦予測モデル - DeBERTa + LightGBM アンサンブル
# ===================================================================
# このコードは、衣服のレビューデータから「レビュワーがその商品を推薦するか」を
# 予測する機械学習モデルを構築します。
# 
# アプローチ:
# 1. DeBERTaの事前予測結果を使用
# 2. テキスト埋め込み（multilingual-e5-large）による特徴抽出
# 3. 多様な特徴量エンジニアリング（感情分析、TF-IDF、統計的特徴）
# 4. LightGBMによる最終予測
# ===================================================================

# ライブラリのインポート
from tqdm.auto import tqdm  # 進捗バー表示
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoModel, AutoTokenizer  # Hugging Face Transformers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ===================================================================
# データ読み込み
# ===================================================================

# 衣服マスターデータの読み込み
cloth_df = pd.read_csv("/kaggle/input/atmacup17dataset/atmacup17_dataset/clothing_master.csv")

# DeBERTaモデルによる事前予測結果を読み込み
# これにより、既に高精度な特徴量として利用可能
train_df = pd.read_csv("/kaggle/input/debert-predictions005/deberta-large-seed0-Ver1-train-predictions (005).csv")
test_df = pd.read_csv("/kaggle/input/debert-predictions005/deberta-large-seed0-Ver1-test-predictions (005).csv")

# ===================================================================
# テキスト埋め込み特徴量の生成
# ===================================================================

# 多言語対応のE5-Large埋め込みモデルを使用
MODEL_ID = "intfloat/multilingual-e5-large"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# トークナイザーとモデルの初期化
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)

# テキストデータの前処理
# タイトルとレビューテキストを結合して統一されたテキスト形式を作成
train_df["text"] = "TITLE: " + train_df["Title"].fillna("none") + " [SEP] " + "Review Text: " + train_df["Review Text"].fillna("none")
test_df["text"] = "TITLE: " + test_df["Title"].fillna("none") + " [SEP] " + "Review Text: " + test_df["Review Text"].fillna("none")

# トークン長の確認（モデルの最大長制限を考慮）
train_max_length = train_df["text"].map(lambda x: len(tokenizer(x)["input_ids"])).max()
test_max_length = test_df["text"].map(lambda x: len(tokenizer(x)["input_ids"])).max()
print(f"最大トークン長 - Train: {train_max_length}, Test: {test_max_length}")

class EmbDataset(Dataset):
    """テキスト埋め込み用のカスタムデータセット"""
    
    def __init__(self, texts, max_length=192):
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, ix):
        # テキストをトークン化し、パディング・切り詰めを実行
        token = self.tokenizer(
            self.texts[ix], 
            max_length=self.max_length, 
            padding="max_length",
            truncation=True,
            return_token_type_ids=True
        )
        return {
            "input_ids": torch.LongTensor(token["input_ids"]),
            "attention_mask": torch.LongTensor(token["attention_mask"]),
            "token_type_ids": torch.LongTensor(token["token_type_ids"])
        }

# 埋め込みベクトルの生成
embeddings = {}

for key, df in zip(["train", "test"], [train_df, test_df]):
    emb_list = []
    dataset = EmbDataset(df["text"].values, max_length=192)
    data_loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=0,
        shuffle=False,
    )
    
    # バッチ処理でテキスト埋め込みを生成
    bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for iter_i, batch in bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        with torch.no_grad():
            # モデルの最後の隠れ層から埋め込みを抽出
            last_hidden_state, pooler_output, hidden_state = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                return_dict=False
            )
            # 平均プーリングで文書レベルの埋め込みを作成
            batch_embs = last_hidden_state.mean(dim=1)

        emb_list.append(batch_embs.detach().cpu().numpy())
    
    embeddings[key] = np.concatenate(emb_list)

# ===================================================================
# 特徴量エンジニアリング
# ===================================================================

# 1. 衣服カテゴリデータの前処理
oe = OrdinalEncoder()
cloth_df[['Division Name', 'Department Name', 'Class Name']] = oe.fit_transform(
    cloth_df[['Division Name', 'Department Name', 'Class Name']]
).astype(int)

# 衣服マスターデータをメインデータにマージ
train_df = train_df.merge(cloth_df, how="left", on="Clothing ID")
test_df = test_df.merge(cloth_df, how="left", on="Clothing ID")

# 2. 感情分析特徴量の追加
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def calculate_sentiment_scores(text):
    """VADER感情分析によるポジティブ/ニュートラル/ネガティブスコア算出"""
    sentiment = sia.polarity_scores(text)
    return sentiment['pos'], sentiment['neu'], sentiment['neg']

# 感情スコアの計算
train_df['sent_pos'], train_df['sent_neu'], train_df['sent_neg'] = zip(*train_df['text'].map(calculate_sentiment_scores))
test_df['sent_pos'], test_df['sent_neu'], test_df['sent_neg'] = zip(*test_df['text'].map(calculate_sentiment_scores))

# 3. テキスト統計特徴量
def calculate_word_count(text):
    """単語数カウント"""
    return len(text.split())

def calculate_char_count(text):
    """文字数カウント"""
    return len(text)

train_df['word_count'] = train_df['text'].map(calculate_word_count)
train_df['char_count'] = train_df['text'].map(calculate_char_count)
test_df['word_count'] = test_df['text'].map(calculate_word_count)
test_df['char_count'] = test_df['text'].map(calculate_char_count)

# 4. ポジティブ/ネガティブワードカウント特徴量
positive_words = ['good', 'great', 'excellent', 'love', 'wonderful', 'best', 'amazing', 'perfect']
negative_words = ['bad', 'terrible', 'poor', 'hate', 'awful', 'worst', 'disappointing', 'horrible']

def count_positive_words(text):
    return sum(word in text.lower() for word in positive_words)

def count_negative_words(text):
    return sum(word in text.lower() for word in negative_words)

train_df['positive_word_count'] = train_df['text'].map(count_positive_words)
train_df['negative_word_count'] = train_df['text'].map(count_negative_words)
test_df['positive_word_count'] = test_df['text'].map(count_positive_words)
test_df['negative_word_count'] = test_df['text'].map(count_negative_words)

# 5. TF-IDF特徴量
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDFベクトライザーの設定（最大500特徴量、1-2グラム）
tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))

# TF-IDFベクトルの生成
tfidf_train = tfidf_vectorizer.fit_transform(train_df['text'])
tfidf_test = tfidf_vectorizer.transform(test_df['text'])

# データフレームに変換して統合
tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_train.shape[1])])
tfidf_test_df = pd.DataFrame(tfidf_test.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_test.shape[1])])

train_df = pd.concat([train_df, tfidf_train_df], axis=1)
test_df = pd.concat([test_df, tfidf_test_df], axis=1)

# 6. 逆説接続詞特徴量
concessive_conjunctions = ["but", "however", "though", "although", "yet", "nevertheless", 
                          "nonetheless", "still", "even though", "even so", "on the other hand"]

def contains_concessive_conjunction(text):
    """逆説の接続詞の有無をチェック"""
    return int(any(conj in text for conj in concessive_conjunctions))

train_df['contains_conjunction'] = train_df['text'].map(contains_concessive_conjunction)
test_df['contains_conjunction'] = test_df['text'].map(contains_concessive_conjunction)

# 7. 埋め込みベクトルの統合
train_df = pd.concat([train_df, pd.DataFrame(embeddings["train"], 
                     columns=[f"emb_{i}" for i in range(embeddings["train"].shape[1])])], axis=1)
test_df = pd.concat([test_df, pd.DataFrame(embeddings["test"], 
                    columns=[f"emb_{i}" for i in range(embeddings["test"].shape[1])])], axis=1)

# 8. レーティング相互作用特徴量の生成
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 特徴量の分類
numeric_features = [
    'Positive Feedback Count', 'Age', 'sent_pos', 'sent_neu', 'sent_neg', 
    'word_count', 'char_count', 'positive_word_count', 'negative_word_count', 'contains_conjunction'
]

categorical_features = ['Division Name', 'Department Name', 'Class Name']
embedding_features = [col for col in train_df.columns if col.startswith('emb_')]
tfidf_features = [col for col in train_df.columns if col.startswith('tfidf_')]

selected_features = numeric_features + embedding_features + tfidf_features + categorical_features

# レーティング予測モデルの構築（テストデータ用の疑似レーティング生成）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features + embedding_features + tfidf_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

rating_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# モデル訓練と予測
rating_model_pipeline.fit(train_df[selected_features], train_df['Rating'])
test_df['predicted_rating'] = rating_model_pipeline.predict(test_df[selected_features])

# レーティング×フィードバック相互作用特徴量
test_df['rating_feedback_interaction'] = test_df['predicted_rating'] * test_df['Positive Feedback Count']
train_df['rating_feedback_interaction'] = train_df['Rating'] * train_df['Positive Feedback Count']

# ===================================================================
# LightGBMモデルの構築と訓練
# ===================================================================

# LightGBMハイパーパラメータの設定
lgb_params = {
    "objective": "binary",  # 二値分類
    "metric": "auc",  # AUC評価
    "learning_rate": 0.1,
    "verbosity": -1,
    "boosting_type": "gbdt",
    "lambda_l1": 0.3,  # L1正則化
    "lambda_l2": 0.3,  # L2正則化
    "max_depth": 6,
    "num_leaves": 128,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "seed": 42,
}

# 特徴量の選択（除外する列を指定）
except_cols = ["Review Text", "Title", "text", "labels", "Recommended IND", "Rating", "prompt"]
features = [col for col in train_df.columns if col not in except_cols]

# 5フォールド層化交差検証の設定
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# 予測結果格納用配列の初期化
oof = np.zeros(train_df.shape[0])  # Out-of-fold予測
preds = np.zeros(test_df.shape[0])  # テストデータ予測

# 交差検証による学習と予測
for fold_ix, (trn_, val_) in enumerate(skf.split(train_df, train_df["labels"])):
    print(f"=== Fold {fold_ix + 1} ===")
    
    # データ分割
    trn_x = train_df.loc[trn_, features]
    trn_y = train_df.loc[trn_, "labels"]
    val_x = train_df.loc[val_, features]
    val_y = train_df.loc[val_, "labels"]

    # LightGBMデータセット作成
    trn_data = lgb.Dataset(trn_x, label=trn_y)
    val_data = lgb.Dataset(val_x, label=val_y)

    # モデル訓練
    lgb_model = lgb.train(
        lgb_params,
        trn_data,
        valid_sets=[trn_data, val_data],
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # アーリーストッピング
            lgb.log_evaluation(100)  # 評価ログ
        ]
    )

    # 予測の保存
    oof[val_] = lgb_model.predict(val_x)
    preds += lgb_model.predict(test_df[features]) / skf.n_splits

# ===================================================================
# 結果評価とサブミッション作成
# ===================================================================

# 交差検証スコア（AUC）の計算
cv_score = roc_auc_score(train_df["labels"], oof)
print(f"CV Score (AUC): {cv_score:.6f}")

# サブミッションファイルの作成
submission_df = pd.DataFrame({"target": preds})
submission_df.to_csv('submission022.csv', index=False)

print("サブミッションファイル作成完了: submission022.csv")
print(f"予測形状: {preds.shape}")
print("予測値のサンプル:")
print(submission_df.head())

# ===================================================================
# 特徴量重要度の可視化（オプション）
# ===================================================================
"""
import matplotlib.pyplot as plt

# 最後のフォールドのモデルから特徴量重要度を取得
feature_importance = lgb_model.feature_importance(importance_type='gain')
feature_names = features

# 上位20特徴量の可視化
indices = np.argsort(feature_importance)[::-1][:20]
plt.figure(figsize=(10, 8))
plt.title("Top 20 Feature Importance")
plt.barh(range(20), feature_importance[indices][::-1])
plt.yticks(range(20), [feature_names[i] for i in indices[::-1]])
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
"""
