# Hướng dẫn chạy hệ thống CTSV News

Tài liệu này mô tả cách chạy đầy đủ bộ:

- Desktop Tool tại `STEP1_ScrapingData\desktop_tool`
- Backend FastAPI tại `STEP6_Dashboard\backend`
- Frontend Dashboard tại `STEP6_Dashboard\frontend`

## 1. Kiến trúc dữ liệu đã chuẩn hóa

### Nguồn dữ liệu local của scraper

- File: `F:\project02-2526\STEP1_ScrapingData\data\posts.db`
- Được tạo bởi: `STEP1_ScrapingData\scraper.py`
- Vai trò: dữ liệu raw local để Desktop Tool đọc và sync

### Nguồn dữ liệu dashboard local

- File hiện tại: `F:\project02-2526\STEP6_Dashboard\backend\ctsv_dashboard.db`
- Vai trò: database chuẩn cho backend và frontend local dev

### Frontend đọc ở đâu

Frontend không đọc trực tiếp file SQLite.

Frontend chỉ gọi Backend API tại:

- `http://localhost:8010/api`

Flow chuẩn:

`Facebook -> scraper local -> posts.db -> Desktop Tool sync -> backend DB -> frontend dashboard`

Trong flow này, dữ liệu sẽ được:

1. ẩn danh ở Desktop Tool bằng `STEP2_Anonymize\Anonymize_CRF.py` trước khi gửi lên backend
2. backend tự gán nhãn AI sau khi ingest thành công

## 2. Port local dev đã chốt

- Backend API: `8010`
- Frontend dev: `4173`
- Frontend preview: `4174`

## 3. Chạy Backend

```powershell
cd F:\project02-2526\STEP6_Dashboard\backend
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
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

## 4. Chạy Frontend

```powershell
cd F:\project02-2526\STEP6_Dashboard\frontend
npm.cmd install
npm.cmd run dev
```

Mở:

- `http://localhost:4173`

## 5. Chạy Desktop Tool

```powershell
cd F:\project02-2526\STEP1_ScrapingData\desktop_tool
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.main
```

Trong app, cấu hình đúng:

- `Project folder`: `F:\project02-2526\STEP1_ScrapingData`
- `scraper.py`: `F:\project02-2526\STEP1_ScrapingData\scraper.py`
- `requirements`: `F:\project02-2526\STEP1_ScrapingData\requirements-scraper-core.txt`
- `posts.db`: `F:\project02-2526\STEP1_ScrapingData\data\posts.db`
- `cookies.json`: `F:\project02-2526\STEP1_ScrapingData\cookies.json`
- `Backend API`: `http://localhost:8010/api`
- `API token`: `ctsv-demo-desktop-token`

Lưu ý:

- Desktop Tool lưu config tại `%APPDATA%\CTSVScraperTool\config.json`
- Nếu trước đó app đã lưu URL cũ như `http://localhost:8000/api`, cần sửa lại rồi bấm lưu

## 6. Chạy crawl và đồng bộ

### Bước 1. Chuẩn bị cookie

- Dùng nút `Lấy cookie Facebook` trong Desktop Tool
- Hoặc đặt `cookies.json` đúng vào `STEP1_ScrapingData`

### Bước 2. Chạy scraper

Trong Desktop Tool:

- bấm `Chạy cào dữ liệu`

Sau khi chạy, dữ liệu local sẽ nằm trong:

- `F:\project02-2526\STEP1_ScrapingData\data\posts.db`

### Bước 3. Đồng bộ lên backend

Trong Desktop Tool:

- bấm `Đồng bộ ngay`

Khi thành công:

- backend ghi dữ liệu vào `ctsv_dashboard.db`
- backend tự gọi AI provider để gắn topic, sentiment và label cảm xúc
- frontend dashboard sẽ đọc số liệu mới qua API

## 7. Kiểm tra dashboard sau khi sync

Đăng nhập dashboard tại:

- `http://localhost:4173`

Tài khoản:

- `admin@example.com`
- `Admin@123456`

Kiểm tra các trang:

- `Tổng quan`
- `Quản lý tin`
- `Phân tích AI`
- `Báo cáo`

Label cảm xúc hiện tại:

- `0`: tiêu cực
- `1`: trung lập
- `2`: tích cực

## 8. Deploy VPS bằng Docker Compose

Port tránh trùng đã chốt:

- PostgreSQL host: `55432`
- Backend host: `18000`
- Frontend host: `18080`

Không dùng:

- `3307`
- `8081`
- `9090`

Chạy:

```bash
cd /opt/project02-2526/deploy
cp .env.example .env
docker compose up -d --build
```

Truy cập:

- Backend API: `http://SERVER_IP:18000`
- Swagger: `http://SERVER_IP:18000/docs`
- Frontend: `http://SERVER_IP:18080`

Desktop Tool trên máy CTSV sẽ trỏ tới:

- `http://SERVER_IP:18000/api`

## 9. Kết luận

Nguồn dữ liệu chuẩn hiện tại đã chốt như sau:

- Scraper local ghi vào `STEP1_ScrapingData\data\posts.db`
- Desktop Tool đọc `posts.db` và sync lên backend
- Backend lưu vào `STEP6_Dashboard\backend\ctsv_dashboard.db` trong local dev
- Frontend chỉ đọc dữ liệu từ Backend API
