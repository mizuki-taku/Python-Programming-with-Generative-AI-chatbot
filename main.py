import os
import gspread
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime  
from zoneinfo import ZoneInfo
from oauth2client.service_account import ServiceAccountCredentials
from file_loader import load_pdf
from file_loader import load_text
from file_loader import load_docx
from text_splitter import split_text
from faiss_indexer import load_and_index_folder, search_index, create_faiss_index

# 環境変数のロード（必要に応じて）
load_dotenv()

# Streamlitのヘッダー
st.title("質問応答チャットボット（情報システム実験）")

# フォルダのパス
lecture_folder = "./information-systems-experiment"  # 講義資料フォルダ
example_folder = "./information-systems-experiment_example"     # 回答例フォルダ
log_folder = "./logs"             # 会話ログ保存フォルダ

# フォルダの存在確認
folders_to_load = [lecture_folder]
if os.path.exists(example_folder):
    folders_to_load.append(example_folder)

# インデックスの作成
def load_and_index_multiple_folders(folders):
    all_texts = []
    for folder in folders:
        texts = load_and_index_folder(folder, return_documents=True)
        all_texts.extend(texts)
    return create_faiss_index(all_texts)
    
# Google Sheets に接続
def get_gsheet():
    import json
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GSPREAD_SERVICE_ACCOUNT"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open(st.secrets["SHEET_NAME"]).sheet1
    return sheet
    
# 会話履歴を1行だけGoogle Sheetsに保存（student_idは常にanonymous）
def save_single_turn_to_sheet(user_query, assistant_response, student_id, student_name):
    sheet = get_gsheet()
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, student_id, student_name, user_query, assistant_response])
    
#過去１０件の履歴を取得
def fetch_recent_history_text(student_id: str, limit: int = 10) -> str:
    """指定 student_id の直近Q/Aを最大 limit 件だけ取得し、古→新で整形"""
    if not student_id:
        return ""

    sheet = get_gsheet()
    rows = sheet.get_all_values()
    if not rows:
        return ""

    # ヘッダ列の位置（なければデフォルト位置を使用）
    header = rows[0]
    def col(name, default):
        return header.index(name) if name in header else default

    col_ts  = col("timestamp", 0)
    col_sid = col("student_id", 1)
    col_q   = col("question", 3)
    col_a   = col("answer", 4)

    pairs = []
    for r in rows[1:]:
        try:
            ts = datetime.strptime(r[col_ts], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Tokyo"))
        except Exception:
            continue

        if r[col_sid].strip() != (student_id or "").strip():
            continue

        q = r[col_q] if len(r) > col_q else ""
        a = r[col_a] if len(r) > col_a else ""
        if q or a:
            pairs.append((ts, q, a))

    # 新しい順に並べて先頭 limit 件を取得 → 表示は古い→新しいに戻す
    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = pairs[:limit]
    pairs.sort(key=lambda x: x[0])

    lines = [f"[{ts.strftime('%Y-%m-%d %H:%M')}] Q: {q}\nA: {a}" for ts, q, a in pairs]
    return "\n\n".join(lines)

    
# フォルダをまとめて読み込み＆インデックス化
combined_index = load_and_index_multiple_folders(folders_to_load)

# セッションステートでメッセージの履歴を保持
if "messages" not in st.session_state:
    st.session_state.messages = []

# 既存のメッセージをブラウザ上に表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Moodleからパラメータ受け取り ---
params = st.experimental_get_query_params()
student_id = params.get("student_id", [None])[0]
student_name= params.get("student_name", [None])[0]

# 学生情報を画面に表示（デバッグ用）
if "student_info_shown" not in st.session_state:
    st.session_state["student_info_shown"] = True
    st.info(f"ようこそ {student_name} さん (学籍番号: {student_id})")
    
# ユーザー入力
query = st.chat_input("質問を入力してください:")

# 質問が入力された場合の応答処理
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ▼ 追加：当日＋昨日の履歴テキストを作成
    history_text = fetch_recent_history_text(student_id)

    # ▼ 変更：履歴をプロンプトに渡す
    response = search_index(combined_index, query, history_text=history_text)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # ここでログを即時保存
    save_single_turn_to_sheet(query, response, student_id, student_name)
