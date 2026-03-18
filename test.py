from google import genai
import os

# Nhớ thay bằng Key mới của Duy nhé
API_KEY = "AIzaSyDRE7sbdinMy30yE3M4vuGiqAsRMIYBe-I" 
client = genai.Client(api_key=API_KEY)

print("--- Danh sách các mô hình khả dụng ---")
for model in client.models.list():
    # SDK mới sử dụng các thuộc tính đơn giản hơn
    print(f"ID: {model.name}")
    print(f"Tên hiển thị: {model.display_name}")
    print("-" * 30)