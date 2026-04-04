import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("🔍 Đang quét danh sách model trên Groq cho Duy...\n")

try:
    models = client.models.list()
    print(f"{'MODEL ID':<40} | {'OWNED BY':<15}")
    print("-" * 60)
    for m in models.data:
        print(f"{m.id:<40} | {m.owned_by:<15}")
except Exception as e:
    print(f"❌ Lỗi: {e}")