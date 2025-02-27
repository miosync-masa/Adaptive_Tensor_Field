import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from openai import OpenAI
from upstash_redis import Redis
import uvicorn
import json
import os
import uuid
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 初期化
app = FastAPI()

# ✅ CORSミドルウェアの適用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番運用時は "https://www.miosync.link" に限定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 環境変数の取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")

# OpenAI クライアントの初期化
client = OpenAI(api_key=OPENAI_API_KEY)

# Upstash Redis クライアント
redis_client = Redis(
    url=UPSTASH_REDIS_URL,
    token=UPSTASH_REDIS_TOKEN
)

# データモデル
class DataItem(BaseModel):
    informationName: str
    category: str
    summary: str
    certification: str

class SearchQuery(BaseModel):
    query: str

# 🔥 1. エンベディング生成関数
def generate_embedding(text: str):
    """OpenAI API でテキストのエンベディングを生成"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 🔥 2. GPT-4o で検索用タグを生成
def generate_search_tags(query: str):
    """ユーザー入力をもとに、スピルリナやバイオシリカに関するタグ＋説明文を生成"""
    system_prompt = (
        "あなたはスピルリナやバイオシリカの専門家です。\n"
        "ユーザーの検索意図を理解し、関連する **タグ** と **簡単な説明文** を3〜5個生成してください。\n"
        "出力は以下のJSON形式のみを返してください。\n\n"
        "例: \n"
        '{ "tags": [\n'
        '  {"tag": "#スピルリナ", "description": "スーパーフードとしてのスピルリナの栄養価と健康効果"},\n'
        '  {"tag": "#免疫力向上", "description": "スピルリナが免疫系に与える影響"},\n'
        '  {"tag": "#サプリメント", "description": "健康補助食品としてのスピルリナの活用法"}\n'
        "]}\n\n"
        "他のテキストは絶対に含めないでください。"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            max_tokens=4000
        )

        logger.info(f"📝 GPTのレスポンス: {response}")

        if response and response.choices:
            tags_json = json.loads(response.choices[0].message.content.strip())
            return tags_json.get("tags", [])

        return []
    
    except Exception as e:
        logger.error(f"❌ タグ生成エラー: {e}")
        return []

# 🔥 3. データ登録 (`POST /api/addData`)
@app.post("/api/addData")
async def add_data(data: DataItem):
    try:
        data_id = str(uuid.uuid4())
        updated_at = datetime.utcnow().isoformat()

        # OpenAI APIでエンベディング生成（title + summary）
        embedding = generate_embedding(f"{data.informationName}. {data.summary}")

        # データを Redis に保存
        redis_client.set(data_id, json.dumps({
            "id": data_id,
            "title": data.informationName,
            "category": data.category,
            "summary": data.summary,
            "metadata": {
                "updated_at": updated_at,
                "certification": data.certification
            },
            "embedding": embedding
        }))

        return {"message": "データが登録されました！", "id": data_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🔥 4. タイトル一覧を取得 (`GET /api/getTitles`)
@app.get("/api/getTitles")
async def get_titles(category: str = Query(None, description="絞り込みカテゴリー（オプション）")):
    try:
        keys = redis_client.keys("*")
        titles = []

        for key in keys:
            data = json.loads(redis_client.get(key))
            if category and data.get("category") != category:
                continue
            titles.append({
                "id": key,
                "title": data.get("title"),
                "metadata": data.get("metadata")
            })

        return {"titles": titles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🔥 6. 特定のデータを取得 (`GET /api/getData`)
@app.get("/api/getData")
async def get_data(data_id: str):
    try:
        existing_data = redis_client.get(data_id)
        if not existing_data:
            raise HTTPException(status_code=404, detail="データが見つかりません")

        return json.loads(existing_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🔥 6. IDでデータを更新 (`PUT /api/updateData`)
@app.put("/api/updateData")
async def update_data(data_id: str, data: DataItem):
    try:
        print(f"🔍 更新リクエスト: data_id={data_id}")  # デバッグ用ログ

        existing_data = redis_client.get(data_id)
        if not existing_data:
            print(f"⚠️ データが見つからない！ data_id={data_id}")  # ログ
            raise HTTPException(status_code=404, detail="データが見つかりません")

        updated_at = datetime.utcnow().isoformat()

        # OpenAI APIでエンベディングを再生成
        embedding = generate_embedding(f"{data.informationName}. {data.summary}")

        # 更新データを Redis に保存
        redis_client.set(data_id, json.dumps({
            "id": data_id,
            "title": data.informationName,
            "category": data.category,
            "summary": data.summary,
            "metadata": {
                "updated_at": updated_at,
                "certification": data.certification
            },
            "embedding": embedding
        }))

        return {"message": "データが更新されました！", "id": data_id}
    except Exception as e:
        print(f"⚠️ 更新エラー: {e}")  # デバッグ用
        raise HTTPException(status_code=500, detail=str(e))

# 🔥 7. タイトルでデータを更新 (`PUT /api/updateDataByTitle`)
@app.put("/api/updateDataByTitle")
async def update_data_by_title(title: str, data: DataItem):
    try:
        print(f"🔍 タイトルでの更新リクエスト: title={title}")  # デバッグ用ログ

        keys = redis_client.keys("*")
        data_id = None

        for key in keys:
            record = json.loads(redis_client.get(key))
            if record.get("title") == title:
                data_id = key
                break

        if not data_id:
            print(f"⚠️ タイトルが見つからない！ title={title}")  # デバッグログ
            raise HTTPException(status_code=404, detail="タイトルが見つかりません")

        updated_at = datetime.utcnow().isoformat()

        # OpenAI APIでエンベディングを再生成
        embedding = generate_embedding(f"{data.informationName}. {data.summary}")

        # 更新データを Redis に保存
        redis_client.set(data_id, json.dumps({
            "id": data_id,
            "title": data.informationName,
            "category": data.category,
            "summary": data.summary,
            "metadata": {
                "updated_at": updated_at,
                "certification": data.certification
            },
            "embedding": embedding
        }))

        return {"message": "データが更新されました！", "id": data_id}
    except Exception as e:
        print(f"⚠️ タイトル更新エラー: {e}")  # デバッグ用
        raise HTTPException(status_code=500, detail=str(e))

# 🔥 5. 検索API (`POST /api/search`)
@app.post("/api/search")
async def search_data(request: SearchQuery):
    try:
        query = request.query
        logger.info(f"🔍 受信した検索クエリ: {query}")

        if not query:
            raise HTTPException(status_code=400, detail="検索クエリが空です")

        # ① GPT-4o で検索用タグを生成
        tags_data = generate_search_tags(query)
        logger.info(f"🏷️ 生成されたタグデータ: {tags_data}")

        if not tags_data:
            raise HTTPException(status_code=500, detail="タグの生成に失敗しました")

        # **💡 タグ + 説明文を一つのテキストに結合**
        enhanced_query = " | ".join([f"{t['tag']} - {t['description']}" for t in tags_data])
        logger.info(f"🔍 検索用エンベディング入力: {enhanced_query}")

        # **💡 OpenAI API に適したフォーマットで送信**
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[enhanced_query]
        )

        # **✅ 修正: 正しいエンベディング取得方法**
        embedding = embedding_response.data[0].embedding
        logger.info(f"🔢 生成されたエンベディング: {embedding[:5]} ... (省略)")

        # ③ Redis に保存されたデータとベクトル検索
        keys = redis_client.keys("*")
        results = []
        logger.info(f"📌 Redis に保存されているデータ数: {len(keys)}")

        for key in keys:
            data = json.loads(redis_client.get(key))
            stored_embedding = data.get("embedding", [])

            logger.info(f"📄 検索対象: {data.get('title')}, カテゴリ: {data.get('category')}")
            if stored_embedding:
                similarity = sum(a * b for a, b in zip(embedding, stored_embedding))
                logger.info(f"📊 類似度: {similarity}")

                if similarity > 0.65:  # 類似度しきい値
                    results.append({
                        "id": key,
                        "title": data.get("title"),
                        "category": data.get("category"),
                        "summary": data.get("summary"),
                        "metadata": data.get("metadata"),
                        "similarity": similarity
                    })
                    logger.info(f"✅ {data.get('title')} を検索結果に追加！（類似度: {similarity}）")
                else:
                    logger.info(f"❌ {data.get('title')} は類似度が低すぎた（{similarity}）")

        # 🔥 類似度順にソート（上位3件のみを取得）
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:3]  # **ここで上位3件に制限！**
        
        logger.info(f"🎯 検索結果数: {len(top_results)} 件（上位3件のみ返却）")
        return {"results": top_results}

    except Exception as e:
        logger.error(f"❌ 検索中にエラー発生: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def root():
    return {"message": "FastAPI is running successfully!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
