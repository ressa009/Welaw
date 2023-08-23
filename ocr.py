import asyncio
import cv2
from dotenv import load_dotenv
import easyocr
import requests
import numpy as np
import openai
from quart import jsonify
import os
import json
from asyncgpt import asyncgpt
from chatbot_db import DBManager
from network_utils import send_response


load_dotenv()
openai.api_key = os.getenv('GPT_API_KEY')


initial_response = {
    "version": "2.0",
    "useCallback": True,
}


# DB 연결
db = DBManager(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT')),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    db=os.getenv('DB_NAME')
)


# GPT에서 값 분류해서 얻는 함수
async def classified_info(ocr_result):

    prompt_1 = " 아래 {}내의 값은 easyocr을 통해 인식한 명함의 텍스트야. 명함에 포함된 내용에 대해 설명해줄게. 내 설명에 따라 명함의 요소를 분류해줘."

    prompt_2 = """
                 회사명은 한글과 영어가 모두 있다면 한글로 분류해줘.
                 이름은 2글자~4글자이고 이,김,최,박,윤,남,류,곽,강,신,손,장,현,위,나,문,정,조,임,한,오,서,황,인,송,전,홍,유,고,문 등으로 시작해. 띄어쓰기가 되어 있을수도 있어.
                 직급은 [사원, 대리, 주임, 차장, 과장, 부장, 센터장, 이사, 대표이사] 등이 있어. 비슷한 단어가 있다면 내가 제시한 목록의 값으로 교정해줘.
                 연락처, 전화번호, hp 등 여러 내용이 있다면 010 으로 시작하는 개인 번호만 알려줘. 만약 010으로 시작하는 번호가 없다면, 지역번호가 포함된 대표 번호나, 연락처를 알려주면 돼. 단 fax 번호는 알려주지 않아도 되고, 여러개의 번호를 반환하지마. 
                 이메일은 E와 같이 첫 대문자를 해줘. 소문자로 표현된 주소만 반환해줘. 그리고 .com / .kr / .net 등으로 끝나니까 .표시가 없다면 넣어줘.

                 분류된 값을 다음과 같이 작성해서 반환해줘. 단, 이름에 띄어쓰기가 포함돼 있다면 없애줘.
                 {
                     "이름": ,
                     "회사명": ,
                     "직급": ,
                     "주소": ,
                     "전화번호": ,
                     "이메일":
                 }

                 단, 프로그램 코드로 작성하지 말고, 위의 텍스트에 대응되는 결과만 알려주면 돼.
                 """

    messages = []

    bot = asyncgpt.GPT(apikey=openai.api_key) # bot 초기화
    messages.append({"role": "user", "content": prompt_1 + ocr_result + prompt_2})

    completion = await bot.chat_complete(messages)


    return str(completion)



# OCR 수행 함수 
def perform_ocr(img_path):

    THRESHOLD=0.3

    image_nparray = np.asarray(bytearray(requests.get(img_path).content), dtype=np.uint8)
    img = cv2.imdecode(image_nparray, cv2.IMREAD_GRAYSCALE)

    reader = easyocr.Reader(['ko', 'en'], gpu=True)

    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    ret, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result = reader.readtext(img)

    r = []

    for bbox, text, conf in result:
        if conf > THRESHOLD:
            r.append(text)

    return " ".join(r)



# Easyocr 비동기 수행을 위해 개별 루프를 얻는 함수 
async def call_ocr_and_send_response(img_path, callback_url, user_id):

    loop = asyncio.get_running_loop()
    ocr_result = await loop.run_in_executor(None, perform_ocr, img_path)

    classified_text = await classified_info(ocr_result)
    print("classified_text: ", classified_text)
    classified_result = json.loads(classified_text)

    print("classified_result: ", classified_result)

    for info in classified_result:

        if classified_result.get(info) in ['', None, 'Null', 'null']:
            print(info)
            classified_result[info] = '-'


    await db.insert_ocr_result_into_db(user_id, classified_result)


    response = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "itemCard": {
                        "thumbnail": {
                            "imageUrl": img_path,
                            "width": 800,
                            "height": 400
                        },
                        "profile": {
                            "title": "OCR 인식 결과",
                            "imageUrl": "https://talk.kakaocdn.net/dna/MtPA5/bl58VFbDBnL/NnlE8YE2lLxABlFXWD9tnY/i_72e7d8aceb86.png?credential=zf3biCPbmWRjbqf40YGePFLewdou7TIK&expires=1785569573&signature=TdP6tilWywE6gvRk0yfWdL%2Fcqv4%3D"
                        },
                        "itemList": [
                            {
                                "title": "이름",
                                "description": classified_result.get("이름")
                            },
                            {
                                "title": "회사명",
                                "description": classified_result.get("회사명")
                            },
                            {
                                "title": "직급",
                                "description": classified_result.get("직급")
                            },
                            {
                                "title": "주소",
                                "description": classified_result.get("주소")
                            },
                            {
                                "title": "전화번호",
                                "description": classified_result.get("전화번호")
                            },
                            {
                                "title": "이메일",
                                "description": classified_result.get("이메일")
                            }
                        ],
                    "itemListAlignment" : "right",
                    "simpleText": {
                            "text": "수정하시려면 아래 수정 항목에 해당하는 버튼을 눌러주세요."
                        }
                }
                }
            ],
            "quickReplies": [
              {
                "messageText": "이름 수정",
                "action": "message",
                "label": "이름 수정"
              },
              {
                "messageText": "회사명 수정",
                "action": "message",
                "label": "회사명 수정"
              },
              {
                "messageText": "직급 수정",
                "action": "message",
                "label": "직급 수정"
              },
              {
                "messageText": "주소 수정",
                "action": "message",
                "label": "주소 수정"
              },
              {
                "messageText": "전화번호 수정",
                "action": "message",
                "label": "전화번호 수정"
              },
              {
                "messageText": "이메일 수정",
                "action": "message",
                "label": "이메일 수정"
              }
            ]
        }
    }

    await send_response(callback_url, response)  # 콜백 URL에 결과를 비동기적으로 전송


# controller 값을 전달받아서 콜백url 유효 시간을 연장하는 함수
async def get_image(user_request):
    if user_request.get('userRequest',{}).get('user',{}).get('properties',{}).get('isFriend') is not True:
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

    img_path = user_request.get('action', {}).get('params', {}).get('business_card')
    print("img_path: ", img_path)


    asyncio.create_task(call_ocr_and_send_response(img_path, callback_url, user_id))

    return jsonify(initial_response)




# db에 접근 및 내용 수정
async def perform_ocr_modify(request_type, modification, user_id):

    db_column_name_map = {
        "이름 수정": "name",
        "회사명 수정": "company_name",
        "직급 수정": "position",
        "주소 수정": "address",
        "전화번호 수정": "phone_number",
        "이메일 수정": "email"
    }

    db_column_name = db_column_name_map.get(request_type)
    print(modification, request_type, db_column_name)

    if db_column_name is not None:
        # await
        await db.update_ocr_result_in_db(user_id, db_column_name, modification)

    response = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"{request_type}이 완료 되었습니다."
                    }
                }
            ],
            "quickReplies": [
              {
                "messageText": "이름 수정",
                "action": "message",
                "label": "이름 수정"
              },
              {
                "messageText": "회사명 수정",
                "action": "message",
                "label": "회사명 수정"
              },
              {
                "messageText": "직급 수정",
                "action": "message",
                "label": "직급 수정"
              },
              {
                "messageText": "주소 수정",
                "action": "message",
                "label": "주소 수정"
              },
              {
                "messageText": "전화번호 수정",
                "action": "message",
                "label": "전화번호 수정"
              },
              {
                "messageText": "이메일 수정",
                "action": "message",
                "label": "이메일 수정"
              }
            ]
    }}

    return response



# Easyocr 비동기 수행을 위해 개별 루프를 얻는 함수 
async def call_modify_and_send_response(request_type, modification, user_id, callback_url):

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, asyncio.run, perform_ocr_modify(request_type, modification, user_id))

    await send_response(callback_url, response)  # 콜백 URL에 결과를 비동기적으로 전송




# ocr 인식 결과 수정
async def get_modify(user_request):

    user_id = user_request.get('userRequest', {}).get('user', {}).get('properties', {}).get('plusfriendUserKey')
    print("user_id : ", user_id)
    callback_url = user_request.get('userRequest', {}).get('callbackUrl')

    request_type = user_request.get('userRequest', {}).get('utterance')
    print("수정요청 : ", request_type)

    modification = user_request.get('action', {}).get('detailParams', {}).get('modified_value',{}).get('origin')
    print("수정내용 : ", modification)

    asyncio.create_task(call_modify_and_send_response(request_type, modification, user_id, callback_url))

    return jsonify(initial_response)