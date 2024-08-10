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

# Настройка логирования
logging.basicConfig(level=logging.INFO)

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

def safe_markdown_format(text):
    # Экранируем специальные символы Markdown
    escape_chars = '_*[]()~`>#+-=|{}.!'
    escaped_text = ''.join(f'\\{char}' if char in escape_chars else char for char in text)
    
    # Дополнительно обрабатываем символ '#', который может быть проблематичным
    escaped_text = escaped_text.replace('#', '\\#')
    
    return escaped_text
    
def escape_markdown(text):
    escape_chars = '_*[]()~`>#+-=|{}.!'
    return ''.join(f'\\{char}' if char in escape_chars else char for char in text)

async def generate_response(prompt):
    try:
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=750
        )
        content = response.choices[0].message.content
        # Применяем безопасное форматирование к полученному контенту
        safe_content = safe_markdown_format(content)
        return safe_content
    except Exception as e:
        logging.error(f"Ошибка при обращении к Groq API: {e}")
        return safe_markdown_format("Извините, произошла ошибка при генерации ответа.")

def search_similar_chunks(query, index, chunks, k=7):
    query_vector = embedding_model.encode([query])
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]
    
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(escape_markdown("Привет! Я бот, который может помочь с информацией о Swisstronik. Используйте следующие команды:\n"
                         "/ctx <запрос> - для поиска по контексту\n"
                         "/ctxsum <запрос> - для суммаризации контекста\n"
                         "/ts <текст> - для перевода текста на русский"), parse_mode="MarkdownV2")

@dp.message(Command("ctx"))
async def cmd_ctx(message: types.Message):
    query = message.text.replace("/ctx", "").strip()
    if not query:
        await message.answer(safe_markdown_format("Пожалуйста, задайте вопрос после команды /ctx"), parse_mode="MarkdownV2")
        return

    await message.answer(safe_markdown_format("Ищу информацию и формирую ответ..."), parse_mode="MarkdownV2")

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
        await message.answer(response[i:i+max_length], parse_mode="MarkdownV2")

    await message.answer(safe_markdown_format("Если у вас есть дополнительные вопросы или нужны уточнения, не стесняйтесь спрашивать!"), parse_mode="MarkdownV2")
    
@dp.message(Command("ctxsum"))
async def cmd_ctxsum(message: types.Message):
    query = message.text.replace("/ctxsum", "").strip()
    if not query:
        await message.answer(escape_markdown("Пожалуйста, укажите запрос после команды /ctxsum"), parse_mode="MarkdownV2")
        return

    relevant_chunks = search_similar_chunks(query, index, chunks)
    context = "\n".join([chunk['content'] for chunk in relevant_chunks])
    
    summary_prompt = f"Summarize the following context about Swisstronik, related to the query: {query}\n\nContext:\n{context}"
    summary = await generate_response(summary_prompt)
    
    await message.answer(escape_markdown(f"Суммаризация контекста для запроса '{query}':"), parse_mode="MarkdownV2")
    await message.answer(summary, parse_mode="MarkdownV2")

@dp.message(Command("ts"))
async def cmd_translate(message: types.Message):
    text_to_translate = message.text.replace("/ts", "").strip()
    if not text_to_translate and message.reply_to_message:
        text_to_translate = message.reply_to_message.text
    
    if not text_to_translate:
        await message.answer(escape_markdown("Пожалуйста, укажите текст для перевода после команды /ts или ответьте на сообщение с текстом"), parse_mode="MarkdownV2")
        return

    try:
        result = deepl_translator.translate_text(text_to_translate, target_lang="RU")
        await message.answer(escape_markdown(f"Перевод:\n{result.text}"), parse_mode="MarkdownV2")
    except Exception as e:
        logging.error(f"Ошибка при переводе: {e}")
        await message.answer(escape_markdown("Извините, произошла ошибка при переводе текста."), parse_mode="MarkdownV2")

@dp.message()
async def message_handler(message: types.Message) -> None:
    # Генерируем ответ на основе входящего сообщения
    response = await generate_response(message.text)
    await message.answer(response, parse_mode="MarkdownV2")
    await message.answer(escape_markdown("Вы также можете использовать команды /ctx или /ctxsum для работы с контекстом об Anthropic, или /ts для перевода."), parse_mode="MarkdownV2")

async def main():
    logging.info("Запуск бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
