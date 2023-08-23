import sys
from quart import Quart, request
from gpt_question import question
from ocr import get_image, get_modify
from pdf import law_question, search_question


application = Quart(__name__)


# 시나리오2 - GPT 호출 함수
@application.route("/question", methods=["POST"])
async def call_gpt():

    user_request = await request.json
    print(user_request)

    return await question(user_request)


# 시나리오3 - OCR 호출 함수
@application.route("/ocr", methods=["POST"])
async def call_ocr():

    user_request = await request.json
    print(user_request)

    return await get_image(user_request)


# 시나리오3 - OCR 인식 결과 수정 함수
@application.route("/ocr_modify", methods=["POST"])
async def call_modify():

    user_request = await request.json
    print(user_request)

    return await get_modify(user_request)


# 시나리오4 - PDF 호출 함수
@application.route("/pdf", methods=["POST"])
async def call_pdf():

    user_request = await request.json
    print(user_request)

    return await law_question(user_request)

# 시나리오 - PDF 검색 함수 호출
@application.route("/pdf_search", methods=["POST"])
async def call_search():

    user_request = await request.json
    print(user_request)

    return await search_question(user_request)


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=int(sys.argv[1]))