import os
import json
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Получение токенов из переменных окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Проверка наличия всех необходимых токенов
if not all([TELEGRAM_TOKEN, GROQ_API_KEY]):
    raise ValueError("Отсутствуют необходимые переменные окружения")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Инициализация клиента Groq
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Пути к файлам
index_path = "anthropic_embeddings.index"
chunks_path = "chunks.json"

# Загрузка модели для эмбеддингов
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)

# Загрузка индекса FAISS и чанков
def load_faiss_index():
    return faiss.read_index(index_path)

def load_chunks():
    with open(chunks_path, 'r', encoding='utf-8') as f:
        return json.load(f)

index = load_faiss_index()
chunks = load_chunks()

async def generate_response(prompt):
    """Генерация ответа с использованием Groq API"""
    try:
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=750
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Ошибка при обращении к Groq API: {e}")
        return "Извините, произошла ошибка при генерации ответа."

def search_similar_chunks(query, index, chunks, k=3):
    query_vector = embedding_model.encode([query])
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]

async def process_query(query):
    relevant_chunks = search_similar_chunks(query, index, chunks)
    context = "\n".join(relevant_chunks)
    prompt = f"Based on the following context about Anthropic, answer the question: {query}\n\nContext:\n{context}"
    response = await generate_response(prompt)
    return response, relevant_chunks

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот, который может ответить на вопросы об Anthropic. Просто задайте свой вопрос, и я постараюсь на него ответить.")

@dp.message()
async def message_handler(message: types.Message) -> None:
    user_message = message.text

    await message.answer("Обрабатываю ваш запрос...")

    try:
        response, relevant_chunks = await process_query(user_message)
        
        await message.answer("Сгенерированный ответ:")
        await message.answer(response)
        
        chunks_text = "\n\n".join([f"Чанк {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
        await message.answer("Релевантные чанки:")
        await message.answer(chunks_text)
    except Exception as e:
        logging.error(f"Ошибка при обработке сообщения: {e}")
        await message.answer("Извините, произошла ошибка при обработке вашего запроса.")

async def main():
    logging.info("Запуск бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
