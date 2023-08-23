import aiomysql
import json

class DBManager:
    def __init__(self, host, port, user, password, db):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db

    async def run_db_query(self, query, args=()):
        conn = await aiomysql.connect(host=self.host, port=self.port,
                                      user=self.user, password=self.password,
                                      db=self.db)
        cur = await conn.cursor()
        await cur.execute(query, args)
        result = await cur.fetchall()
        await conn.commit()
        await cur.close()
        conn.close()
        return result

    async def get_messages_from_db(self, session_id):
        query = "SELECT messages FROM sessions WHERE session_id = %s"
        result = await self.run_db_query(query, (session_id,))
        if result:
            messages_str = result[0][0]
            messages = json.loads(messages_str)
        else:
            messages = []
        return messages

    async def update_messages_in_db(self, session_id, messages):
        messages_str = json.dumps(messages)
        query = "REPLACE INTO sessions (session_id, messages) VALUES (%s, %s)"
        await self.run_db_query(query, (session_id, messages_str))

    async def insert_ocr_result_into_db(self, user_id, result):
        query = """
        INSERT INTO ocr_table (user_id, name, company_name, position, address, phone_number, email)
        VALUES (%s, %s, %s, %s, %s, %s, %s) AS ocr_result
        ON DUPLICATE KEY UPDATE
            name = ocr_result.name,
            company_name = ocr_result.company_name,
            position = ocr_result.position,
            address = ocr_result.address,
            phone_number = ocr_result.phone_number,
            email = ocr_result.email
        """
        args = (user_id, result.get("이름"), result.get("회사명"), result.get("직급"),
                result.get("주소"), result.get("전화번호"), result.get("이메일"))
        await self.run_db_query(query, args)

    async def update_ocr_result_in_db(self, user_id, modify_request, modification):
        query = f"UPDATE ocr_table SET {modify_request} = %s WHERE user_id = %s"
        await self.run_db_query(query, (modification, user_id))

    async def get_email_address(self, user_id):
        query = "SELECT email FROM ocr_table WHERE user_id = %s"
        result = await self.run_db_query(query, (user_id,))
        if result:
            return result[0][0]
        else:
            return None
