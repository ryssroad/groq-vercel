# api/telegram_webhook.py

import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from tavily import TavilyClient
from ai import generateText
from ai_sdk.openai import createOpenAI

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Получение токенов из переменных окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
WEBHOOK_PATH = os.getenv('WEBHOOK_PATH', '/api/telegram_webhook')

# Проверка наличия всех необходимых токенов
if not all([TELEGRAM_TOKEN, TAVILY_API_KEY, GROQ_API_KEY]):
    raise ValueError("Отсутствуют необходимые переменные окружения")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Инициализация клиентов
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
groq = createOpenAI({
    'baseURL': 'https://api.groq.com/openai/v1',
    'apiKey': GROQ_API_KEY,
})

async def generate_response(prompt):
    try:
        response = await generateText({
            'model': groq('mixtral-8x7b-32768'),
            'prompt': prompt,
        })
        return response.text
    except Exception as e:
        logging.error(f"Ошибка при обращении к Groq API: {e}")
        return "Извините, произошла ошибка при генерации ответа."

@dp.message(Command("summary"))
async def command_summary_handler(message: Message) -> None:
    query = message.text.replace("/summary", "").strip()
    
    if not query:
        await message.answer("Пожалуйста, укажите тему для поиска новостей после команды /summary")
        return

    await message.answer("Ищу и суммирую последние новости по вашему запросу...")
    
    try:
        news_results = tavily_client.search(query=query, search_depth="advanced", include_images=False, max_results=5)
        
        summary_prompt = "Summarize the following news results in Russian:\n\n"
        for result in news_results.get('results', []):
            summary_prompt += f"Title: {result.get('title', 'No title')}\nContent: {result.get('content', 'No content')}\n\n"
        
        summary = await generate_response(summary_prompt)
        
        formatted_summary = f"Краткая сводка новостей по запросу '{query}':\n\n{summary}"
        
        await message.answer(formatted_summary)
    except Exception as e:
        logging.error(f"Ошибка при выполнении суммаризации новостей: {e}")
        await message.answer("Извините, произошла ошибка при обработке вашего запроса.")

@dp.message(Command("ask"))
async def command_ask_handler(message: Message) -> None:
    query = message.text.replace("/ask", "").strip()
    
    if not query:
        await message.answer("Пожалуйста, задайте вопрос после команды /ask")
        return

    await message.answer("Ищу ответ на ваш вопрос...")
    
    try:
        answer = tavily_client.qna_search(query=query)
        formatted_answer = f"Ответ на ваш вопрос:\n\n{answer}"
        await message.answer(formatted_answer)
    except Exception as e:
        logging.error(f"Ошибка при выполнении быстрого поиска: {e}")
        await message.answer("Извините, произошла ошибка при обработке вашего вопроса.")

@dp.message(Command("search"))
async def command_search_handler(message: Message) -> None:
    query = message.text.replace("/search", "").strip()
    
    if not query:
        await message.answer("Пожалуйста, укажите запрос для поиска после команды /search")
        return

    await message.answer("Выполняю поиск...")
    
    try:
        search_results = tavily_client.search(query=query, search_depth="basic", include_images=False, max_results=5)
        
        links = f"Результаты поиска по запросу '{query}':\n\n"
        for i, result in enumerate(search_results.get('results', []), 1):
            links += f"{i}. {result.get('title', 'Без заголовка')}\n"
            links += f"🔗 {result.get('url', 'Нет ссылки')}\n\n"
        
        await message.answer(links)
    except Exception as e:
        logging.error(f"Ошибка при выполнении поиска: {e}")
        await message.answer("Извините, произошла ошибка при обработке вашего запроса.")

@dp.message()
async def message_handler(message: types.Message) -> None:
    user_message = message.text

    try:
        response = await generate_response(user_message)
        await message.answer(response)
    except Exception as e:
        logging.error(f"Ошибка при обработке сообщения: {e}")
        await message.answer("Извините, произошла ошибка при обработке вашего запроса.")

async def handle_webhook(request):
    update = types.Update(**await request.json())
    await dp.feed_update(bot, update)

from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class WebhookHandler(BaseHTTPRequestHandler):
    async def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        update = types.Update(**json.loads(post_data))
        
        await dp.feed_update(bot, update)
        
        self.send_response(200)
        self.end_headers()

def main():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, WebhookHandler)
    print('Starting server...')
    httpd.serve_forever()

if __name__ == "__main__":
    main()
