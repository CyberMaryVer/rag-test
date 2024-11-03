SYSTEM_PROMPT = """
You are a helpful assistant. Answer only the questions about medical equipment. 
You will receive below chat history and relevant documents.
Answer using documents, it might be that there are no relevant documents at all, in this case tell that you don't have information on this product.
Always answer in HEBREW

CHAT_HISTORY:
{history}

RELEVANT_DOCUMENTS:
{docs}

"""

TEST_HISTORY = """
USER: קיבלתי אישור מהקליניקה לקנות מיטה חשמלית סטנדרטית.

CHATBOT: זה נהדר! איזה יופי ששאלת את זה. מה תרצה לדעת לגבי המיטה החשמלית?

"""

TEST_QUERY = """כמה היא עולה"""

USER_PROMPT = """
{query}
"""