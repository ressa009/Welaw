from asyncgpt import asyncgpt
import asyncio
from transformers import GPT2Tokenizer
from quart import jsonify
from chatbot_db import DBManager
from network_utils import send_response
from dotenv import load_dotenv
import os

load_dotenv()


db = DBManager(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT')),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    db=os.getenv('DB_NAME')
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

async def get_answer(user_utterance, callback_url, session_id):
    bot = asyncgpt.GPT(apikey=os.getenv('GPT_API_KEY'))

    messages = await db.get_messages_from_db(session_id)  # 세션 데이터를 데이터베이스에서 불러오기
    messages.append({"role": "user", "content": user_utterance}) # 유저 발화 저장
    tokens = tokenizer.encode(str(messages)) # 토큰 수 세기
    if len(tokens) >= 6500 or len(messages) > 20: # 적절한 수준으로 토큰 유지
        messages.pop(0)
        if len(tokens) >= 7500:
            messages.pop(0)

    await db.update_messages_in_db(session_id, messages)  # 데이터를 데이터베이스에 저장

    completion = await bot.chat_complete(messages)
    answer = str(completion)
    messages.append({"role": "system", "content": answer})
    tokens = tokenizer.encode(str(messages))
    # 토큰 재확인
    if len(tokens) >= 6500 or len(messages) > 20:
        messages.pop(0)

    await db.update_messages_in_db(session_id, messages)  # 데이터를 데이터베이스에 저장

    response = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ],
            "quickReplies": [
                  {
                    "messageText": "계속 질문하기",
                    "action": "message",
                    "label": "계속 질문하기"
                  },
                  {
                    "messageText": "다른 서비스 이용",
                    "action": "message",
                    "label": "다른 서비스 이용"
                  }
                ]
        }
    }
    # 답변 보내는 함수 호출
    print(response)
    await send_response(callback_url, response)

async def question(user_request):
    # 플러스친구인지 확인해서, 친구가 아니면 친구요청
    if user_request.get('userRequest', {}).get('user', {}).get('properties', {}).get('isFriend') is not True:
        fail_response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "채널 추가 후 이용 부탁드립니다."
                        }
                    }
                ]
            }
        }
        return jsonify(fail_response)

    user_id = user_request.get('userRequest', {}).get('user', {}).get('properties', {}).get('plusfriendUserKey', '')
    callback_url = user_request.get('userRequest', {}).get('callbackUrl')
    print("callback_url: ", callback_url)
    user_utterance = user_request.get('action', {}).get('params', {}).get('question_contents')# 파라미터로 넘어온 변수명 주의
    print("user_utterance: ", user_utterance)

    initial_response = {
        "version": "2.0",
        "useCallback": True,
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "콜백"
                    }
                }
            ],
            }
    }

    # 비동기 함수를 직접 호출하지 않고, asyncio.create_task를 사용하여 백그라운드에서 실행
    asyncio.create_task(get_answer(user_utterance, callback_url, user_id))

    return jsonify(initial_response)
