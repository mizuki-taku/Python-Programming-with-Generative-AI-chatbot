import os
import openai
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from file_loader import load_pdf, load_text, load_docx
from text_splitter import split_text

# .envファイルの内容を読み込み
load_dotenv()

# 環境変数からAPIキー、モデル、温度を取得
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_TEMPERATURE  = float(os.getenv("OPENAI_API_TEMPERATURE", "0.7"))
OPENAI_API_MAX_TOKENS   = int(os.getenv("OPENAI_API_MAX_TOKENS",   "1000"))
OPENAI_API_TOP_K        = int(os.getenv("OPENAI_API_TOP_K",        "3"))
EMBEDDING_MODEL_NAME    = os.getenv("OPENAI_EMBEDDING_MODEL",      "text-embedding-3-small")
OPENAI_API_MODEL        = os.getenv("OPENAI_API_MODEL",            "gpt-4o")

# 埋め込みとインデックス作成
def create_faiss_index(texts):
    """
    Document群をベクトル化してFAISSインデックスを構築
    Embeddingモデルは環境変数 OPENAI_EMBEDDING_MODEL で指定
    """
    embeddings = OpenAIEmbeddings(
        api_key=openai.api_key,
        model=EMBEDDING_MODEL_NAME
    )
    return FAISS.from_documents(texts, embeddings)
    

def search_docs(faiss_index, query):
    """FAISSで検索して上位チャンクを返す"""
    return faiss_index.similarity_search(query, k=OPENAI_API_TOP_K)

# クエリ検索とChatGPT 4.0oでの応答生成
def search_index(faiss_index, query, history_pairs=None):
    """
    history_pairs: [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    """

    # FAISSインデックスで検索
    results = faiss_index.similarity_search(query)
    content = results[0].page_content if results else "該当する情報が資料内に見つかりませんでした。"

    # system プロンプト（役割定義）
    system_message = {
        "role": "system",
        "content": (
                #学生の質問やコメントに対する基本的な役割とアプローチ
                "あなたは講義資料に基づき、生成AIを活用したPython学習をサポートする正確かつ簡潔な回答を行うアシスタントです。"
                "丁寧語で回答し、感謝の言葉や『先生の回答：』といった接頭辞は含めないでください。"
                
                #範囲外・不確定な情報への対応
                "講義の範囲（特定のライブラリや手法）を超えた技術質問には、その旨を伝えつつ、関連する一般的なPythonの書き方やAI活用のコツを簡潔に補足してください。"
                "感想が資料に直接関係なくても、AIによる効率化やプログラミングの楽しさに関連付けて共感・補足を行ってください。"
                "特定のAIモデルの内部仕様など不確定な情報は避け、Pythonの文法や標準的なライブラリの知識で補完してください。"
                
                #抽象的な発言に対する具体化支援
                "「コードが動かない」「AIの回答が変」等の抽象的な悩みには、『思いつきで大丈夫ですよ』と添えて、AIへの指示（プロンプト）を具体化する問いかけを行ってください。"
                "1.【入力の明確化】：『AIにどんな指示（プロンプト）を出しましたか？あるいは、どんなエラーメッセージが出ていますか？』"
                "2.【処理の分解】：『そのプログラムをいくつかのステップに分けるとしたら、最初はどの部分を動かしたいですか？』"
                "3.【期待する動作】：『実行した結果、画面にはどのような表示が出るのが理想ですか？』"
                
                #伴走型支援と問いかけ（AI活用特化）
                "回答は必ず『共感』や『肯定』から始め、最後は学生が「次の一手」を試せるような問いかけを1つ添えてください。"
                "1.【コードの理解】：『AIが生成したこの部分のコードの中で、意味が少し分かりにくいと感じる行はありますか？』"
                "2.【応用・実験】：『もし、〇〇という機能を追加したい場合、AIにどんな追加指示を出せば良さそうですか？』"
                "3.【デバッグの練習】：『エラーを直すために、まずは print文で中身を確認してみませんか？興味はありますか？』"
                "学生が『全然わからない』場合も、AIとの対話のきっかけを一緒に整理する優しい姿勢を保ってください。"
                
                #履歴の活用と生成言語
                "会話履歴を前提知識とし、以前のエラーや作成中のコードの文脈を踏まえてアドバイスしてください。"
                "回答言語は質問（主要部分）の言語に合わせてください。"
                
                #授業コメントへの反応方針
                "AIを使うことへの戸惑いや発見に寄り添い、技術を道具として使いこなす姿勢をポジティブに励ます回答を提供してください。"

        )
    }

    messages = [system_message]

    # 🔽 ここが最重要：履歴を「会話」として追加
    if history_pairs:
        messages.extend(history_pairs)

    # 🔽 今回の質問
    messages.append({
        "role": "user",
        "content": (
            f"【参考資料】\n{content}\n\n"
            f"質問: {query}"
        )
    })

    response = openai.chat.completions.create(
        model=OPENAI_API_MODEL,
        temperature=OPENAI_API_TEMPERATURE,
        max_tokens=OPENAI_API_MAX_TOKENS,
        messages=messages,
    )

    return response.choices[0].message.content


def load_and_index_folder(folder_path, return_documents=False):
    all_texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # ✅ 追加：空ファイルをスキップ
        if os.path.getsize(file_path) == 0:
            print(f"スキップ（空ファイル）: {filename}")
            continue

        if filename.endswith(".pdf"):
            documents = load_pdf(file_path)
        elif filename.endswith(".txt"):
            documents = load_text(file_path)
        elif filename.endswith(".docx"):
            documents = load_docx(file_path)
        else:
            continue

        texts = split_text(documents)
        all_texts.extend(texts)

    if return_documents:
        return all_texts
    return create_faiss_index(all_texts)

