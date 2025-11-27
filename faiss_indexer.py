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
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")
OPENAI_API_TEMPERATURE = float(os.getenv("OPENAI_API_TEMPERATURE"))
OPENAI_API_MAX_TOKENS = int(os.getenv("OPENAI_API_MAX_TOKENS"))
OPENAI_API_TOP_K = int(os.getenv("OPENAI_API_TOP_K"))
EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL")

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
def search_index(faiss_index, query, history_text: str = ""):
    # FAISSインデックスで検索
    results = faiss_index.similarity_search(query)
    content = results[0].page_content if results else "該当する情報が資料内に見つかりませんでした。"

    # ChatGPT 4.0oを使って応答を生成
    response = openai.chat.completions.create(
    model=OPENAI_API_MODEL,
    temperature=OPENAI_API_TEMPERATURE,
    max_tokens=OPENAI_API_MAX_TOKENS,
    messages=[
        {
            "role": "system",
            "content": (
                # 学生の質問やコメントに対する基本的な役割とアプローチ
                "あなたは学生の質問や授業に関するコメントに対して、提供された講義資料と参考例を基に講義の範囲に基づいて正確で簡潔な回答を行うアシスタントです。"
                "質問、コメントに対しては、基本的に丁寧語で回答してください。"
                "感謝の言葉などは必要ありません"
                "先生の回答:の部分も必要ありません"

                # 講義の範囲を超えた質問に対する対応方法
               "質問が明らかに講義の範囲を超えている場合には、その旨を伝えつつ、関連する一般的な知識や補助的な情報があれば簡潔に紹介してください。"
                "感想が資料に直接関係していない場合でも、可能な範囲で講義内容やテーマと関連づけて、共感や補足情報を添えて反応してください。"
                
                #過去の履歴の使用方法
                "履歴は参考としてのみ使用し、回答本文に履歴の羅列・引用・要約は含めないでください。"
                "過去の質問と今回が類似している場合は、重複説明を避け、前回からの不足点や更新点を優先して補足してください。"
                
                # 授業コメントへの反応の方針
                "授業に関するコメントに対しては、その内容に基づいて適切に反応し、学生が持っている印象や興味を引き出す回答を提供してください。"
                "回答は、学生の学びや関心を深めるために、教育的かつ励ましのある内容であることが望ましいです。"
        
                # 不確定な情報に対する慎重な対応
                "原則として提供した資料に基づいて回答してください．回答に必要な関連情報が不十分な場合には、要求工学に関する一般的な知識を用いて補完しても構いません。"
                "推測に基づく不確定な情報の提供は行わないでください。"

                #生成言語
                "質問が英語で書かれていた場合は、回答も英語で行ってください。"
                "質問が日本語で書かれていた場合は、日本語で回答してください。"
                "言語が混在している場合は、質問の主要な部分の言語に合わせて回答してください。"
        ),
    },
        {
            "role": "user",
            "content": (
                    "以下の資料に基づいて回答してください。必要に応じて一般知識で補完しても構いません。\n"
                    f"{content}\n\n"
                    "【直近履歴（最大10件）】\n"
                    f"{history_text}\n\n"
                    f"質問: {query}"
            ),
        }
    ]
)

    answer = response.choices[0].message.content.strip()
    return answer


# フォルダ内のすべてのPDFおよびテキストファイルを読み込んでインデックスを作成
def load_and_index_folder(folder_path, return_documents=False):
    all_texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            documents = load_pdf(file_path)
        elif filename.endswith(".txt"):
            documents = load_text(file_path)
        elif filename.endswith(".docx"):  # Wordファイル対応
            documents = load_docx(file_path)
        else:
            continue

        texts = split_text(documents)
        all_texts.extend(texts)

    # return_documentsがTrueの場合、ドキュメントのリストを返す
    if return_documents:
        return all_texts
    # デフォルトではインデックスを返す
    return create_faiss_index(all_texts)

