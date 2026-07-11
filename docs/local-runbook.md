# Local Runbook

## Mục tiêu

Tài liệu này chốt cách chạy local dev chuẩn cho toàn hệ thống CTSV:

- Frontend dashboard chỉ đọc dữ liệu từ Backend API
- Backend là nguồn dữ liệu online chuẩn cho dashboard
- Desktop Tool đọc dữ liệu local từ SQLite scraper rồi đồng bộ lên backend

## 1. Nguồn dữ liệu chuẩn

### Dữ liệu local do scraper tạo

- File: `F:\project02-2526\STEP1_ScrapingData\data\posts.db`
- Vai trò: dữ liệu thô local trên máy CTSV
- Được ghi bởi: `STEP1_ScrapingData\scraper.py`

### Dữ liệu online cho dashboard

- File local dev hiện tại: `F:\project02-2526\STEP6_Dashboard\backend\ctsv_dashboard.db`
- Vai trò: nguồn dữ liệu chuẩn để backend trả API cho frontend
- Được ghi bởi: backend sau khi nhận batch ingest từ Desktop Tool

### Frontend lấy dữ liệu từ đâu

Frontend không đọc trực tiếp `posts.db` hay `ctsv_dashboard.db`.

Frontend chỉ gọi:

- `http://localhost:8010/api`

## 2. Port local dev đã chốt

- Backend API: `8010`
- Frontend dev: `4173`
- Frontend preview: `4174`

## 3. Chạy backend

```powershell
cd F:\project02-2526\STEP6_Dashboard\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

Kiểm tra:

- Health: `http://127.0.0.1:8010/health`
- Swagger: `http://127.0.0.1:8010/docs`

Tài khoản mặc định:

- Email: `admin@example.com`
- Password: `Admin@123456`

Desktop API token mặc định:

- `ctsv-demo-desktop-token`

## 4. Chạy frontend

```powershell
cd F:\project02-2526\STEP6_Dashboard\frontend
npm.cmd run dev
```

Mở:

- `http://localhost:4173`

## 5. Chạy Desktop Tool

```powershell
cd F:\project02-2526\STEP1_ScrapingData\desktop_tool
.\.venv\Scripts\python.exe -m app.main
```

Trong app, cấu hình:

- Backend API URL: `http://localhost:8010/api`
- API token: `ctsv-demo-desktop-token`
- SQLite output: `F:\project02-2526\STEP1_ScrapingData\data\posts.db`

Lưu ý:

- Desktop Tool lưu config thật ở `%APPDATA%\CTSVScraperTool\config.json`
- Nếu trước đó đã lưu port cũ như `8000`, cần sửa lại trong app rồi bấm lưu

## 6. Flow end-to-end chuẩn

1. Chạy backend
2. Chạy frontend và đăng nhập
3. Chạy Desktop Tool
4. Kiểm tra cookie nếu cần crawl
5. Chạy scraper để ghi vào `STEP1_ScrapingData\data\posts.db`
6. Bấm `Đồng bộ ngay` để gửi dữ liệu lên backend
7. Refresh dashboard để xem số liệu mới

## 7. Điều cần tránh

- Không để frontend đọc trực tiếp `posts.db`
- Không để dashboard vừa đọc SQLite local scraper vừa đọc DB backend
- Không dùng `F:\project02-2526\data` như nguồn dữ liệu chính cho flow CTSV hiện tại

## 8. Kết luận

Flow chuẩn hiện tại là:

`Facebook -> scraper local -> posts.db local -> Desktop Tool sync -> backend DB chuẩn -> frontend dashboard`
