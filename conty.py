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
import deepl

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования (если еще не настроено)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Получение токенов из переменных окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')

# Проверка наличия всех необходимых токенов
if not all([TELEGRAM_TOKEN, GROQ_API_KEY, DEEPL_API_KEY]):
    raise ValueError("Отсутствуют необходимые переменные окружения")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Инициализация клиентов
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)
deepl_translator = deepl.Translator(DEEPL_API_KEY)

# Пути к файлам
index_path = "swiss_embeddings.index"
chunks_path = "swiss_chunks.json"

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
            temperature=0.9,
            max_tokens=750
        )
        # Логирование сырого ответа
        logger.info(f"Сырой ответ от Groq API: {response}")

        return response.choices[0].message.content

        # Логирование извлеченного контента
        logger.info(f"Извлеченный контент: {content}")
        
    except Exception as e:
        logging.error(f"Ошибка при обращении к Groq API: {e}")
        return "Извините, произошла ошибка при генерации ответа."

def search_similar_chunks(query, index, chunks, k=7):
    query_vector = embedding_model.encode([query])
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]
    
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот, который может помочь с информацией о Swisstronik. Используйте следующие команды:\n"
                         "/ctx <запрос> - для поиска по контексту\n"
                         "/ctxsum <запрос> - для суммаризации контекста\n"
                         "/ts <текст> - для перевода текста на русский")

@dp.message(Command("ctx"))
async def cmd_ctx(message: types.Message):
    query = message.text.replace("/ctx", "").strip()
    if not query:
        await message.answer("Пожалуйста, задайте вопрос после команды /ctx")
        return

    await message.answer("Ищу информацию и формирую ответ...")

    relevant_chunks = search_similar_chunks(query, index, chunks)
    context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])

    prompt = f"""На основе следующего контекста о Swisstronik, пожалуйста, 
    ответьте на вопрос: "{query}"

    Контекст:
    {context}

    Пожалуйста, дайте подробный и структурированный ответ. Если информации 
    недостаточно, укажите это. Если вопрос касается создания смарт-контракта, 
    предоставьте пошаговое руководство с примерами кода, где это уместно."""

    response = await generate_response(prompt)

    # Разделяем ответ на части и отправляем
    max_length = 4000
    for i in range(0, len(response), max_length):
        await message.answer(response[i:i+max_length])

    await message.answer("Если у вас есть дополнительные вопросы или нужны уточнения, не стесняйтесь спрашивать!")

@dp.message(Command("ctxsum"))
async def cmd_ctxsum(message: types.Message):
    query = message.text.replace("/ctxsum", "").strip()
    if not query:
        await message.answer("Пожалуйста, укажите запрос после команды /ctxsum")
        return

    relevant_chunks = search_similar_chunks(query, index, chunks)
    context = "\n".join([chunk['content'] for chunk in relevant_chunks])
    
    summary_prompt = f"Summarize the following context about Swisstronik, related to the query: {query}\n\nContext:\n{context}"
    summary = await generate_response(summary_prompt)
    
    await message.answer(f"Суммаризация контекста для запроса '{query}':")
    await message.answer(summary)

@dp.message(Command("ts"))
async def cmd_translate(message: types.Message):
    text_to_translate = message.text.replace("/ts", "").strip()
    if not text_to_translate and message.reply_to_message:
        text_to_translate = message.reply_to_message.text
    
    if not text_to_translate:
        await message.answer("Пожалуйста, укажите текст для перевода после команды /ts или ответьте на сообщение с текстом")
        return

    try:
        result = deepl_translator.translate_text(text_to_translate, target_lang="RU")
        await message.answer(f"Перевод:\n{result.text}")
    except Exception as e:
        logging.error(f"Ошибка при переводе: {e}")
        await message.answer("Извините, произошла ошибка при переводе текста.")

@dp.message()
async def message_handler(message: types.Message) -> None:
    # Генерируем ответ на основе входящего сообщения
    response = await generate_response(message.text)
    await message.answer(response)
    await message.answer("Вы также можете использовать команды /ctx или /ctxsum для работы с контекстом об Anthropic, или /ts для перевода.")

async def main():
    logging.info("Запуск бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
