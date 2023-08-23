from asyncgpt.asyncgpt.chatgpt import GPT
import asyncio, httpx, tempfile, os, re, openai, aiosmtplib
from transformers import GPT2Tokenizer
from quart import jsonify
from PyPDF2 import PdfReader
from email.mime.text import MIMEText
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from network_utils import send_response
import numpy as np
import pandas as pd
from chatbot_db import DBManager
from typing import List, Dict
from langchain.schema import AgentAction
from langchain.tools import Tool
from langchain.prompts import StringPromptTemplate

load_dotenv()
bot = GPT(apikey=os.getenv('GPT_API_KEY'))


# DB 연결
db = DBManager(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT')),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    db=os.getenv('DB_NAME')
)


summary = None #전역변수 summary 추가

workspace_path = os.getcwd()
REPOSITORY_PATH = os.path.join(workspace_path, "playground")

embeddings_file_path = os.path.join(REPOSITORY_PATH, 'case_embeddings.npy')
indices_file_path = os.path.join(REPOSITORY_PATH, 'case_indices.npy')

embeddings = np.load(embeddings_file_path).astype('float32')
indices = np.load(indices_file_path)

case_data = pd.read_csv(os.path.join(REPOSITORY_PATH, 'cases.csv'))
sessions = {}
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

async def send_email(recipient, subject, body):
    # 이메일 설정
    sender = os.getenv('EMAIL_ID')
    password = os.getenv('EMAIL_PW')
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender
    message['To'] = recipient

    # 이메일 보내기
    smtp = aiosmtplib.SMTP(hostname="smtp.gmail.com", port=465, use_tls=True)
    await smtp.connect()
    await smtp.login(sender, password)
    await smtp.send_message(message)
    await smtp.quit()


def get_ada_embedding(text):
    return openai.Embedding.create(model="text-embedding-ada-002", input=text)["data"][0]["embedding"]


def split_into_chunks(text, max_length):
    """Split the text into chunks that are at most max_length characters long."""
    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        new_chunk = (current_chunk + " " + sentence.strip()).strip()
        if len(new_chunk) > max_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk = new_chunk
    chunks.append(current_chunk)
    return chunks


async def summarize_chunk(chunk):
    completion = await bot.chat_complete([{"role": "user", "content": f"다음 내용을 한글로 요약해줘.: {chunk}"}])
    return completion.choices[0].message['content']


async def download_and_summarize_pdf(url):
    global summary  # summary를 전역변수로 설정

    # PDF 파일 다운로드
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    response.raise_for_status()  # 응답 성공 보장

    # 임시 파일 생성
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    # 내려받은 내용을 임시 파일에 저장
    with open(temp_file.name, "wb") as f:
        f.write(response.content)

    # PyPDF2로 PDF 파일 읽기
    pdf_reader = PdfReader(temp_file.name)

    # PDF 내용 추출
    text = " ".join([page.extract_text() for page in pdf_reader.pages])

    # 텍스트를 3500자의 청크로 나누기
    chunks = split_into_chunks(text, 3500)

    # 첫 번째 청크에서 사건명 찾기
    prompts = [
        {"role": "user", "content": "판결문 일부: '대법원 1969.2.4. 선고 68다1587 판결\n1\n대법원 1969.2.4.선고 68다1587 판결\n【소유권이전등기말소】, [집17(1)민,155]\n【판시사항】\n사후양자는 호주상속권은 있어도 재산상속권은 없다' 사건명: "},
        {"role": "assistant", "content": "소유권이전등기말소"},
        {"role": "user", "content": "판결문 일부: '서 울 중 앙 지 방 법 원 \n판 결\n사 건 2009고단3458 가. 명예훼손 \n 나. 업무방해\n 피 고 인 1.가.나. 조00 (610619-1000000), MBC 프로듀서 \n 주거 서울 000\n 등록기준지 서산시000' 사건명: "},
        {"role": "assistant", "content": "명예훼손 및 업무방해"},
        {"role": "user", "content": "판결문 일부: '대법원 2018. 4. 12.선고 2017다292244 판결 【소유권확인】, [공2018상,897] 【판시사항】 [1] 부동산 매' 사건명: "},
        {"role": "assistant", "content": "소유권확인"},
        {"role": "user", "content": f"판결문 일부: {chunks[0]} 사건명 : "}
    ]
    case_name_completion = await bot.chat_complete(prompts)
    case_name = case_name_completion.choices[0].message['content']

    tasks = [summarize_chunk(chunk) for chunk in chunks]
    summaries = await asyncio.gather(*tasks)       
    # 요약된 텍스트를 하나의 문자열로 결합
    summary = " ".join(summaries)

    # Remove the temporary file
    os.remove(temp_file.name)
         
    # 사건명과 요약된 텍스트 반환
    return case_name, summary


def find_related_cases(query_embedding):
    k = 2
    query_embedding_np = np.array(query_embedding)  # 리스트를 NumPy 배열로 변환

    # 판결 요지의 인덱스를 찾습니다.
    judgement_indices = case_data['판결요지'].index

    # 임베딩 배열의 크기를 초과하지 않는 인덱스만 선택합니다.
    valid_indices = judgement_indices[judgement_indices < len(embeddings)]

    # 판결 요지의 임베딩만 선택합니다.
    judgement_embeddings = embeddings[valid_indices]

    # 판결 요지의 임베딩만을 사용하여 유사도를 계산합니다.
    similarities = cosine_similarity(judgement_embeddings, query_embedding_np.reshape(1, -1))
    top_k_indices = similarities.flatten().argsort()[-k:][::-1]

    top_k_rows = [case_data.iloc[indices[idx]] for idx in top_k_indices]

    return top_k_rows


async def send_related_cases_email(recipient_email, related_cases):
    email_body = ""
    for case in related_cases:
        email_body += f"【판례일련번호】: {case['판례일련번호']}\n"
        email_body += "\n"
        email_body += "--------------------------------\n"
        email_body += "\n"
        email_body += f"【사건명】: {case['사건명']}\n"
        email_body += "\n"
        email_body += "--------------------------------\n"
        email_body += "\n"
        email_body += f"【사건번호】: {case['사건번호']}\n"
        email_body += "\n"
        email_body += "--------------------------------\n"
        email_body += "\n"
        email_body += f"【사건종류명】: {case['사건종류명']}\n"
        email_body += "\n"
        email_body += "--------------------------------\n"
        email_body += "\n"
        email_body += f"【판시사항】: {case['판시사항']}\n"
        email_body += "\n"
        email_body += "--------------------------------\n"
        email_body += "\n"
        email_body += f"【판결요지】: {case['판결요지']}\n"
        email_body += "\n"
        email_body += "--------------------------------\n"
        email_body += "\n"
        email_body += f"【참조 조문】: {case['참조조문']}\n"
        email_body += "\n"
        email_body += "--------------------------------\n"
        email_body += "\n"
        email_body += f"【참조 판례】: {case['참조판례']}\n"
        email_body += "\n"
        email_body += "--------------------------------\n"
        email_body += "\n"
        email_body += f"【판례내용】: {case['판례내용']}\n"
        email_body += "\n"
        email_body += "--------------------------------\n"
        email_body += "\n"

    # 이메일 보내기
    await send_email(recipient_email, 'Related Cases', email_body)

def find_most_similar_case(query_embedding: np.array, embeddings: np.array) -> Dict:
    most_similar_index = np.argmax(np.dot(query_embedding, embeddings.T))
    case_id = indices[most_similar_index]
    case = case_data.iloc[case_id]
    return case

# def find_most_similar_case(query_embedding: np.array, embeddings: np.array, top_n=5, threshold=0.7) -> Dict:
#     # Calculate similarity scores
#     scores = np.dot(query_embedding, embeddings.T)
    
#     # Get the top N most similar cases
#     top_indices = np.argsort(scores)[-top_n:]
    
#     # Filter out cases with similarity below the threshold
#     top_indices = [idx for idx in top_indices if scores[idx] >= threshold]
    
#     # If no cases meet the threshold, return the most similar one
#     if not top_indices:
#         top_indices = [np.argmax(scores)]
    
#     # Choose one randomly from the top N
#     chosen_index = np.random.choice(top_indices)
    
#     case_id = indices[chosen_index]
#     case = case_data.iloc[case_id]
#     return case

def case_retrieval_func(input):
    # Convert the input to embeddings
    query_embedding = get_embedding(input)
    # Find the most similar case
    case = find_most_similar_case(query_embedding, embeddings)
    return case

tools = [Tool(name="Case Retrieval", description="Retrieve similar cases from the database using embeddings.", func=case_retrieval_func)]

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    input_variables: List[str]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)    

def get_embedding(question: str) -> np.array:
    return get_ada_embedding(question)

async def get_answer(user_utterance, callback_url, session_id):
    global summary  # declare summary as global inside the function

    case_name = ""

    if session_id in sessions:
        messages = sessions[session_id]
    else:
        messages = []

    messages.append({"role": "user", "content": user_utterance})
    tokens = tokenizer.encode(str(messages))
    if len(tokens) >= 6500 or len(messages) > 20:
        messages.pop(0)
        if len(tokens) >= 7500:
            messages.pop(0)
    
    elif user_utterance.startswith('질문'):
        prompts = [{"role": "user", "content": f"{summary}를 바탕으로 {user_utterance}에 대해 답변해줘"},]
        print(prompts)
        completion = await bot.chat_complete(prompts)
        answer = str(completion)
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
                        "action": "message",
                        "label": "서비스안내 처음으로 돌아가기",
                        "messageText": "서비스 안내"
                    },
                    {
                        "action": "message",
                        "label": "서술상담이어가기",
                        "messageText": "📄 서술 상담"
                    }
                ]
            }
        }

    elif user_utterance.startswith('찾기'):
        # '상담' 뒤에 오는 텍스트를 summary에 저장
        summary = user_utterance[2:]

        # 유사한 판례 찾기
        query_embedding = get_embedding(summary)
        case = find_most_similar_case(query_embedding, embeddings)

        # 해당 판례의 요약을 AgentAction으로 생성
        agent_action = AgentAction("Case Retrieval", "", 
            f"비슷한 판례를 보여드리겠습니다.\n판례일련번호 : {case['판례일련번호']}\n사건명 : {case['사건명']}\n판결요지 : \n{case['판결요지']}")

        template = """Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: {input}
        Thought: 
        Action: the action to take, should be one of [{tool_names}]
        Action Input: 
        Observation: 
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: 

        Begin! 

        {agent_scratchpad}
        """

        # Create the prompt template
        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            input_variables=["input", "intermediate_steps"]
        )

        # Fill the template with the case data
        filled_template = prompt.format(
            input=summary,
            intermediate_steps=[
                (agent_action, "End of process"),
                # Add more steps as needed
            ]
        )
        summary = filled_template.split('Begin!')[-1].split('Observation:')[0].strip()
        response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": filled_template.split('Begin!')[-1].split('Observation:')[0].strip() # 수정된 부분
                        }
                    }
                ],
                "quickReplies": [
                    {
                        "action": "message",
                        "label": "서비스안내 처음으로 돌아가기",
                        "messageText": "서비스 안내"
                    },
                    {
                        "action": "message",
                        "label": "상담 이어가기",
                        "messageText": "📃 서술 상담"
                    }
                ]
            }
        }
    elif user_utterance == "📥PDF 업로드":
        answer = "아래 링크를 클릭하여 https://www.file.io/ pdf파일을 업로드 해주세요. 업로드가 완료되면 생성된 URL을 입력해주세요.\n 잠시만 기다려주세요. 1분정도 소요가 됩니다."
        response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer
                        }
                    },
                    {
                        "basicCard": {
                            "buttons": [
                                {
                                    "action": "webLink",
                                    "label": "Go to file.io",
                                    "webLinkUrl": "https://www.file.io/"}]}}]}}
    elif user_utterance.startswith('url :') or user_utterance.startswith('https://file.io/'):
        if user_utterance.startswith('url :'):
            url = user_utterance[5:]
        else:
            url = user_utterance
        case_name, summary = await download_and_summarize_pdf(url)
        answer = f"올려주신 PDF의 요약본입니다.\n사건명: {case_name}\n요약: {summary}"
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
                    "messageText": "유사한 판례 이메일 받기",
                    "action": "message",
                    "label": "유사한 판례 받기"
                  },
                  {
                    "messageText": "유사한 판례 이메일 안 받기",
                    "action": "message",
                    "label": "유사한 판례 안 받기"
                  }
                ]
              }
            }

    elif user_utterance == "유사한 판례 이메일 받기":
        recipient_email = await db.get_email_address(session_id)
        print(recipient_email)
        if recipient_email:
            answer = f"{recipient_email}, 이 주소로 이메일을 보내드릴까요?"
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
                            "messageText": "네, 이메일 주소가 맞습니다.",
                            "action": "message",
                            "label": "네, 이메일 주소가 맞습니다."
                        },
                        {
                            "messageText": "아니요, 이메일 주소가 틀립니다.",
                            "action": "message",
                            "label": "아니요, 이메일 주소가 틀립니다."
                        }
                    ]
                }
            }
        else:
            answer = "사용자 ID에 연결된 이메일 주소를 찾을 수 없습니다. \n 받으려는 이메일 주소를 입력해주세요 \n ex) amugae@test.com"
            response = {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {
                            "simpleText": {
                                "text": answer
                            }
                        }
                    ]
                }
            }

    elif user_utterance == "네, 이메일 주소가 맞습니다.":
        summary_embedding = get_ada_embedding(summary)  # 요약된 판례 텍스트의 임베딩 생성
        related_cases = find_related_cases(summary_embedding)
        recipient_email = await db.get_email_address(session_id)
        asyncio.create_task(send_related_cases_email(recipient_email, related_cases))

        answer = "유사한 판례 정보를 이메일로 보내드렸습니다."
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
                        "action": "message",
                        "label": "서비스안내 처음으로 돌아가기",
                        "messageText": "서비스 안내"
                    },
                    {
                        "action": "message",
                        "label": "PDF 관련 상담하기",
                        "messageText": "📄 서술 상담"
                    }
                ]
            }
        }

    elif user_utterance == "아니요, 이메일 주소가 틀립니다.":
        answer = "이메일을 받으려는 주소를 입력해주세요."
        response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer
                        }
                    }
                ]
            }
        }
    # elif user_utterance == "🔎판례검색":
    #     response = {
    #         "version": "2.0",
    #         "template": {
    #             "outputs": [
    #                 {
    #                     "simpleText": {
    #                         "text": "어떤 방식으로 상담을 원하시나요?"}}],
    #             "quickReplies": [
    #                 {
    #                     "label": "PDF를 업로드합니다",
    #                     "action": "message",
    #                     "messageText": "📥PDF 업로드"
    #                 },
    #                 {
    #                     "label": "직접입력합니다",
    #                     "action": "message",
    #                     "messageText": "🔎직접입력"}]}}
    elif user_utterance == "유사한 판례 이메일 안 받기":
        answer = "이메일을 받지 않는 것을 선택하셨습니다."
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
                        "action": "message",
                        "label": "서비스안내 처음으로 돌아가기",
                        "messageText": "서비스 안내"
                    },
                    {
                        "action": "message",
                        "label": "PDF 관련 상담하기",
                        "messageText": "📄 서술 상담"
                    }
                ]
            }
        }

    # elif user_utterance == "🔎직접입력": 
    #     answer = "'찾기 :'의 뒤에 판례찾기에 필요한 내용을 최대한 자세하게 서술해주세요. 입력 내용을 바탕으로 유사한 판례를 찾아드립니다."
    #     response = {
    #         "version": "2.0",
    #         "template": {
    #             "outputs": [
    #                 {
    #                     "simpleText": {
    #                         "text": answer
    #                     }
    #                 }
    #             ]
    #         }
    #     }

    else:
        # 이메일 주소 패턴 검색
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, user_utterance)
        if email_match:
            recipient_email = email_match.group()
            print(summary)
            summary_embedding = get_ada_embedding(summary)  # 요약된 판례 텍스트의 임베딩 생성
            related_cases = find_related_cases(summary_embedding)
            #case_name_cases = case_data[case_data['사건명'].str.contains(case_name, na=False)]

            asyncio.create_task(send_related_cases_email(recipient_email, related_cases))

            answer = "관련 판례 정보를 이메일로 보내드렸습니다."
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
                            "action": "message",
                            "label": "서비스안내 처음으로 돌아가기",
                            "messageText": "서비스 안내"
                        },
                        {
                            "action": "message",
                            "label": "PDF 관련 상담 이어가기",
                            "messageText": "📄 서술 상담"
                        }
                    ]
                }
            }

        else:
            #sessions[session_id] = messages 
            prompts = [{"role": "user", "content": f"{summary}를 바탕으로 {user_utterance}에 대해 답변해줘"},]
            print(prompts)
            completion = await bot.chat_complete(prompts)
            answer = str(completion)
            messages.append({"role": "system", "content": answer})
            tokens = tokenizer.encode(str(messages))
            print(len(tokens))
            if len(tokens) >= 6500 or len(messages) > 20:
                messages.pop(0)

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
                            "action": "message",
                            "label": "서비스안내 처음으로 돌아가기",
                            "messageText": "서비스 안내"
                        },
                        {
                            "action": "message",
                            "label": "서술상담이어가기",
                            "messageText": "📄 서술 상담"
                        }
                    ]
                }
            }



    await send_response(callback_url, response)


# 컨트롤러에서 호출되는 함수로, 시나리오 2번
async def law_question(user_request):
    
    # 플러스친구인지 확인해서, 친구가 아니면 친구요청 response
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

    if user_request.get('action', {}).get('params', {}).get('summary_pdf'):
        user_utterance = user_request.get('action', {}).get('params', {}).get('summary_pdf')
    elif user_request.get('action', {}).get('params', {}).get('sys_url'):
        user_utterance = user_request.get('action', {}).get('detailParams', {}).get('sys_url',{}).get('origin')
    elif user_request.get('action', {}).get('params', {}).get('email_address'):
        user_utterance = user_request.get('action', {}).get('params', {}).get('email_address')
    # elif user_request.get('action', {}).get('params', {}).get('pdf_search'):
    #     user_utterance = user_request.get('action', {}).get('detailParams', {}).get('pdf_search',{}).get('origin')        
    else:
        user_utterance = user_request.get('userRequest', {}).get('utterance')
        print(user_utterance)
    
    if user_utterance == "🏛️ 법률 상담":
        
        initial_response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "어떤 방식으로 상담을 원하시나요?"}}],
                "quickReplies": [
                    {
                        "label": "PDF로상담받기",
                        "action": "message",
                        "messageText": "📥PDF 업로드"
                    },
                    {
                        "label": "직접입력하여상담받기",
                        "action": "message",
                        "messageText": "🔎직접입력"}]}}

    else:
        
        asyncio.create_task(get_answer(user_utterance, callback_url, user_id))
        initial_response = {
            "version": "2.0",
            "useCallback": True,
        }

    return jsonify(initial_response)

async def search_question(user_request):
    
    # 플러스친구인지 확인해서, 친구가 아니면 친구요청 response
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
    print(user_request)
    user_id = user_request.get('userRequest', {}).get('user', {}).get('properties', {}).get('plusfriendUserKey', '')
    callback_url = user_request.get('userRequest', {}).get('callbackUrl')
    user_utterance = user_request.get('action', {}).get('params', {}).get('search_pdf')
    print(user_utterance)


    if user_utterance == "🐭🙈🐒직접입력": 
        answer = "'찾기 :'의 뒤에 판례찾기에 필요한 내용을 최대한 자세하게 서술해주세요. 입력 내용을 바탕으로 유사한 판례를 찾아드립니다."
        initial_response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer
                        }
                    }
                ]
            }
        }        

    else:
        asyncio.create_task(get_answer(user_utterance, callback_url, user_id))
        initial_response = {
            "version": "2.0",
            "useCallback": True,
        }

    return jsonify(initial_response)

