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
    
def fetch_recent_history_text(student_id: str, limit: int = 10) -> list:
    """指定 student_id の履歴をリスト形式で取得（改行対策済み）"""
    if not student_id:
        return []

    sheet = get_gsheet()
    rows = sheet.get_all_values()
    if len(rows) <= 1:
        return []

    header = rows[0]
    # 列番号の特定（query と response に対応）
    col_sid = header.index("student_id") if "student_id" in header else 1
    col_q = header.index("user_query") if "user_query" in header else 3 # 保存時の名前に合わせる
    col_r = header.index("assistant_response") if "assistant_response" in header else 4

    pairs = []
    # 新しい順にスキャン
    for r in reversed(rows[1:]):
        if len(r) > col_r and r[col_sid].strip() == (student_id or "").strip():
            # QとAをセットにして保存（ここではまだ整形しない）
            pairs.append({"query": r[col_q], "response": r[col_r]})
        if len(pairs) >= limit:
            break

    # 表示用に古い順に戻してリストで返す
    return pairs[::-1]

# Streamlitのヘッダー
st.title("質問応答チャットボット（情報システム実験）")

# --- Moodleからパラメータ受け取り ---
params = st.experimental_get_query_params()
student_id = params.get("student_id", [None])[0]
student_name= params.get("student_name", [None])[0]

# セッションステートでメッセージの履歴を保持
if "messages" not in st.session_state:
    st.session_state.messages = []

    # ▼ 過去10件の会話履歴を取得（修正版：リストが返ってくる）
    history_data = fetch_recent_history_text(student_id, limit=10)

    # 文字列分割をやめ、リストから直接 session_state に入れる
    for item in history_data:
        st.session_state.messages.append({
            "role": "user",
            "content": item["query"]
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": item["response"]
        })

# フォルダのパス
lecture_folder = "./information-network-engineering"  # 講義資料フォルダ
example_folder = "./information-network-engineering-example"     # 回答例フォルダ
log_folder = "./logs"             # 会話ログ保存フォルダ"./logs"

folders_to_load = [lecture_folder]
if os.path.exists(example_folder):
    folders_to_load.append(example_folder)

# フォルダをまとめて読み込み＆インデックス化
combined_index = load_and_index_multiple_folders(folders_to_load)

# --- 既存のメッセージをブラウザ上に表示 ---
# これにより、読み込まれた履歴がチャットUIとして再現されます
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])




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
        st.write(query)

    # ▼ 追加：当日＋昨日の履歴テキストを作成
    response = search_index(
    combined_index,
    query,
    history_pairs=st.session_state.messages[-20:]  # 直近10往復
    )

    with st.chat_message("assistant"):
       st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # ここでログを即時保存
    save_single_turn_to_sheet(query, response, student_id, student_name)
