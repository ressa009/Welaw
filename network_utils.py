import httpx

async def send_response(callback_url, response):

    # 예외 처리를 위한 try-except 블록 추가
    try:
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        async with httpx.AsyncClient() as client:
            res = await client.post(callback_url, json=response, headers=headers)

        # 응답 상태 코드 확인
        if res.status_code == 200:
            print("Response successfully sent")
        else:
            print(f"Failed to send response: {res.status_code}, {res.text}")

    except Exception as e:
        print("An error occurred:", e)