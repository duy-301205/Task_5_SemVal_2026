# SemEval-2026 Task 5:

Dự án tham gia cuộc thi SemEval 2026, tập trung vào việc giải quyết bài toán đánh giá mức độ hợp lí nghĩa của 1 từ đồng âm trong một câu tường thuật sử dụng các mô hình ngôn ngữ lớn (LLM).

# 🚀 Cách tiếp cận (Approach)
Dự án tập trung vào việc tối ưu hóa khả năng suy luận của LLM thông qua các kỹ thuật:

- Prompt Engineering: Thử nghiệm Zero-shot và Few-shot để trích xuất thông tin chính xác.

- Gemini API: Sử dụng dòng mô hình gemini-1.5-flash hoặc gemini-1.5-pro để xử lý ngữ cảnh.

- Evaluation: Đánh giá độ tương quan dựa trên các chỉ số chính thức của SemEval:

-- Spearman Rank Correlation

-- Pearson Correlation

-- MAE (Mean Absolute Error): Xử lý kết quả đầu ra từ API để đảm bảo định dạng đúng yêu cầu của ban tổ chức.

# 📁 Cấu trúc thư mục (Project Structure)
```text
├── data/               # Chứa dữ liệu train/dev/test từ SemEval (.json, .csv)
├── results/            # Lưu trữ kết quả dự đoán và log thực thi chi tiết
├── scripts/            # Các script tự động hóa quy trình chạy experiment
├── src/                # Mã nguồn chính (Core logic)
│   ├── config/         # Cấu hình hệ thống và nạp biến môi trường (dotenv)
│   ├── evaluation/     # Tính toán chỉ số đánh giá (Spearman, Pearson, MAE)
│   ├── models/         # Wrapper cho các LLM (Gemini, Qwen, v.v.)
│   ├── parser/         # Xử lý định dạng dữ liệu đầu vào và đầu ra
│   ├── prompts/        # Quản lý các Template Prompt Engineering
│   └── utils/          # Các hàm tiện ích (Logger, File Helpers)
├── .env                # Biến môi trường (Chứa API Key - ĐÃ BỊ CHẶN BỞI GIT)
├── .gitignore          # Danh sách file loại trừ không đẩy lên GitHub
├── README.md           # Hướng dẫn dự án
└── requirements.txt    # Danh sách thư viện cần thiết

```
# 🛠 Cài đặt & Sử dụng
## 1. Khởi tạo môi trường
Bạn nên sử dụng Python 3.9+ và môi trường ảo .venv:

### Clone dự án
git clone [https://github.com/duy-301205/Task_5_SemVal_2026.git](https://github.com/duy-301205/Task_5_SemVal_2026.git)
cd Task_5_SemVal_2026

### Tạo và kích hoạt môi trường ảo
python -m venv .venv
### Windows:
.venv\Scripts\activate
### Linux/macOS:
source .venv/bin/activate

### Cài đặt thư viện
pip install -r requirements.txt

## 2. Cấu hình API Key
Tạo file .env tại thư mục gốc và thêm key của bạn:

Plaintext
GEMINI_API_KEY=your_api_key_here

## 📊 Kết quả (Experiments)
Hiện tại dự án đang trong quá trình thử nghiệm với mô hình Gemini.

Metric chính: Spearman Correlation, Pearson, MAE.

Trạng thái: Đã hoàn thành baseline với Prompt Zero-shot.
