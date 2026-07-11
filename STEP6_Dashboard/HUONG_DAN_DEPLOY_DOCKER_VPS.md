# Hướng Dẫn Deploy Fullstack Dashboard Lên VPS Bằng Docker

## 1. Mục tiêu triển khai

Hệ thống fullstack gồm 3 thành phần chạy trên VPS:

- `db`: PostgreSQL lưu dữ liệu chính cho dashboard
- `backend`: FastAPI nhận dữ liệu từ Desktop Tool, chạy ẩn danh đã xong từ client, gán nhãn AI, cung cấp API cho frontend
- `frontend`: Dashboard React build ra Nginx để người dùng truy cập qua web

IP public triển khai:

- `103.180.138.225`

Port chuẩn đã chốt để tránh trùng:

- Frontend: `18080`
- Backend: `18000`
- PostgreSQL host: `55432`

## 2. Chuẩn bị VPS

Máy chủ cần có:

- Ubuntu hoặc Debian 64-bit
- Docker Engine
- Docker Compose plugin
- Tối thiểu khuyến nghị:
  - CPU: 4 vCPU
  - RAM: 8 GB
  - Disk: 30 GB trở lên

Cần mở firewall:

- `18080/tcp`
- `18000/tcp`
- `22/tcp`

Không khuyến nghị public trực tiếp `55432` ra Internet nếu không thực sự cần quản trị từ xa.

## 3. Chuẩn bị source code trên VPS

Ví dụ thư mục triển khai:

```bash
/opt/project02-2526
```

Sau khi copy source lên VPS, cần đảm bảo có các thư mục:

```text
/opt/project02-2526/STEP6_Dashboard
/opt/project02-2526/deploy_all/models
```

Thư mục `deploy_all/models` phải chứa ít nhất:

- `phobert/model`
- `simcse/model`
- `bgem3/model`

## 4. Cấu hình file .env

Tại thư mục:

```text
/opt/project02-2526/STEP6_Dashboard/.env
```

Tạo hoặc cập nhật nội dung mẫu:

```env
POSTGRES_DB=ctsv_news
POSTGRES_USER=ctsv
POSTGRES_PASSWORD=doi_mat_khau_manh
DATABASE_URL=postgresql+psycopg2://ctsv:doi_mat_khau_manh@db:5432/ctsv_news
SECRET_KEY=doi_secret_key_rat_dai_va_kho_doan

POSTGRES_HOST_PORT=55432
BACKEND_HOST_PORT=18000
FRONTEND_HOST_PORT=18080

ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=Admin@123456
ADMIN_FULL_NAME=CTSV Administrator

DESKTOP_API_TOKEN=ctsv-demo-desktop-token
CORS_ORIGINS=["http://103.180.138.225:18080","http://localhost:18080"]
VITE_API_BASE_URL=http://103.180.138.225:18000/api

AI_PROVIDER=gemini
GEMINI_API_KEY=thay_bang_api_key_that
GEMINI_MODEL=gemini-1.5-flash

LOCAL_MODELS_DIR=/app/models
ENABLE_LOCAL_MODELS=true
LABEL_ON_INGEST=true
```

Lưu ý:

- Không commit file `.env` thật lên git.
- `GEMINI_API_KEY` là secret, chỉ đặt trên máy local hoặc VPS.
- `VITE_API_BASE_URL` phải là URL public mà frontend gọi được.

## 5. Cách docker-compose hoạt động

File compose nằm tại:

```text
STEP6_Dashboard/docker-compose.yml
```

Backend sẽ mount model local từ repo:

```text
../deploy_all/models:/app/models:ro
```

Nghĩa là trên VPS, thư mục thật phải là:

```text
/opt/project02-2526/deploy_all/models
```

Nếu thiếu thư mục này, cột label local model sẽ để trống hoặc trạng thái `partial/failed`, nhưng hệ thống vẫn có thể chạy với Gemini.

## 6. Chạy build và deploy

Đứng tại:

```bash
cd /opt/project02-2526/STEP6_Dashboard
```

Kiểm tra config trước:

```bash
docker compose config
```

Build và chạy nền:

```bash
docker compose up --build -d
```

Xem trạng thái:

```bash
docker compose ps
```

Xem log backend:

```bash
docker compose logs -f backend
```

Xem log frontend:

```bash
docker compose logs -f frontend
```

Xem log database:

```bash
docker compose logs -f db
```

## 7. Kiểm tra sau deploy

Kiểm tra health backend:

```text
http://103.180.138.225:18000/health
```

Kiểm tra Swagger:

```text
http://103.180.138.225:18000/docs
```

Kiểm tra frontend:

```text
http://103.180.138.225:18080
```

Sau khi login thành công, frontend sẽ gọi backend API, backend đọc dữ liệu từ PostgreSQL rồi trả về dashboard.

## 8. Flow dữ liệu sau khi deploy

Luồng chuẩn:

1. Desktop Tool chạy trên máy CTSV
2. Scraper ghi vào SQLite local `STEP1_ScrapingData/data/posts.db`
3. Desktop Tool đọc SQLite, chạy ẩn danh qua `STEP2_Anonymize/Anonymize_CRF.py`
4. Desktop Tool gửi batch đã ẩn danh lên `http://103.180.138.225:18000/api/ingest/batches`
5. Backend lưu dữ liệu vào PostgreSQL container
6. Backend chạy Gemini + model local và tính `voted_label`
7. Frontend đọc dữ liệu từ backend để hiển thị bảng tin, chart, thống kê

Frontend không đọc trực tiếp SQLite local và cũng không đọc trực tiếp file trong `STEP1_ScrapingData/data`.

## 9. Các lệnh vận hành thường dùng

Restart toàn bộ:

```bash
docker compose restart
```

Restart riêng backend:

```bash
docker compose restart backend
```

Dừng hệ thống:

```bash
docker compose down
```

Dừng và xóa container nhưng giữ dữ liệu volume:

```bash
docker compose down
```

Dừng và xóa cả volume dữ liệu:

```bash
docker compose down -v
```

Chỉ dùng `-v` khi thật sự muốn xóa PostgreSQL data.

## 10. Sao lưu và khôi phục PostgreSQL

Sao lưu:

```bash
docker compose exec db pg_dump -U ctsv -d ctsv_news > backup_ctsv_news.sql
```

Khôi phục:

```bash
cat backup_ctsv_news.sql | docker compose exec -T db psql -U ctsv -d ctsv_news
```

## 11. Lỗi thường gặp

`frontend mở được nhưng không login được`:

- Kiểm tra `VITE_API_BASE_URL`
- Kiểm tra backend có chạy ở `18000` không
- Kiểm tra `CORS_ORIGINS`

`Desktop Tool sync lỗi 401/403`:

- Kiểm tra `DESKTOP_API_TOKEN` trong backend
- Kiểm tra token đang lưu trong Desktop Tool

`cột local model không có nhãn`:

- Kiểm tra thư mục `/opt/project02-2526/deploy_all/models`
- Kiểm tra log backend
- Kiểm tra RAM VPS có đủ để load model không

`Gemini không có nhãn`:

- Kiểm tra `GEMINI_API_KEY`
- Kiểm tra máy chủ có outbound Internet
- Kiểm tra quota hoặc quyền của API key

## 12. Kết luận vận hành

Sau khi deploy bằng Docker:

- Backend, frontend, PostgreSQL đều chạy trên VPS
- Desktop Tool local vẫn là nơi crawl dữ liệu Facebook
- Desktop Tool không ghi trực tiếp vào PostgreSQL
- Mọi dữ liệu đi qua Backend API rồi mới vào database và dashboard
