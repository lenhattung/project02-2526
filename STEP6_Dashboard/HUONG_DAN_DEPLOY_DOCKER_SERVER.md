# Hướng dẫn deploy Docker CTSV Dashboard lên server

Tài liệu này hướng dẫn deploy bộ:

- PostgreSQL
- FastAPI Backend
- React Frontend

trên server public IP:

- `103.180.138.225`

## 1. Kết quả sau khi deploy

Sau khi chạy xong, hệ thống sẽ có:

- Backend API: `http://103.180.138.225:18000`
- Swagger: `http://103.180.138.225:18000/docs`
- Frontend Dashboard: `http://103.180.138.225:18080`

Desktop Tool trên máy CTSV sẽ gửi dữ liệu lên:

- `http://103.180.138.225:18000/api`

## 2. Luồng dữ liệu sau khi deploy

Luồng chuẩn là:

`Facebook -> scraper local trên máy CTSV -> posts.db local -> Desktop Tool sync -> Backend API trên server -> PostgreSQL trong Docker -> Frontend Dashboard đọc từ Backend API`

Điều này có nghĩa là:

- Desktop Tool không ghi trực tiếp vào PostgreSQL
- Desktop Tool gửi dữ liệu qua HTTP API của backend
- Desktop Tool ẩn danh nội dung trước khi gửi lên backend
- Backend nhận batch ingest rồi ghi vào database PostgreSQL trong Docker
- Backend tự gọi AI provider để gắn nhãn sau ingest
- Frontend gọi backend để load và visualize chart data

## 3. Các port đã chọn

Để tránh trùng với các port bạn đã cảnh báo, bộ deploy này dùng:

- PostgreSQL host port: `55432`
- Backend host port: `18000`
- Frontend host port: `18080`

Không dùng các port:

- `3307`
- `8081`
- `9090`

## 4. File deploy trong repo

Các file chính:

- `F:\project02-2526\STEP6_Dashboard\docker-compose.yml`
- `F:\project02-2526\STEP6_Dashboard\.env.example`

## 5. Chuẩn bị trên server

Trên server Linux, cài sẵn:

- Docker
- Docker Compose plugin

Kiểm tra:

```bash
docker --version
docker compose version
```

## 6. Upload source lên server

Ví dụ đặt source tại:

```bash
/opt/project02-2526
```

Sau khi upload, cấu trúc cần có:

```text
/opt/project02-2526/STEP6_Dashboard
```

## 7. Tạo file `.env`

Vào thư mục deploy:

```bash
cd /opt/project02-2526/STEP6_Dashboard
cp .env.example .env
nano .env
```

## 8. Cấu hình `.env`

Ví dụ:

```env
POSTGRES_DB=ctsv_news
POSTGRES_USER=ctsv
POSTGRES_PASSWORD=doi_mat_khau_rat_manh
DATABASE_URL=postgresql+psycopg2://ctsv:doi_mat_khau_rat_manh@db:5432/ctsv_news
SECRET_KEY=doi_secret_key_rat_manh

POSTGRES_HOST_PORT=55432
BACKEND_HOST_PORT=18000
FRONTEND_HOST_PORT=18080

ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=Admin@123456
ADMIN_FULL_NAME=CTSV Administrator

DESKTOP_API_TOKEN=ctsv-demo-desktop-token
CORS_ORIGINS=["http://103.180.138.225:18080"]
VITE_API_BASE_URL=http://103.180.138.225:18000/api
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
```

Khuyến nghị:

- đổi `POSTGRES_PASSWORD`
- đổi `SECRET_KEY`
- đổi `ADMIN_PASSWORD`
- đổi `DESKTOP_API_TOKEN`
- kiểm tra `GEMINI_API_KEY` hợp lệ trước khi chạy ingest thật

## 9. Chạy deploy

```bash
cd /opt/project02-2526/STEP6_Dashboard
docker compose up --build -d
```

Kiểm tra container:

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

## 10. Kiểm tra sau deploy

Mở các URL:

- `http://103.180.138.225:18000/health`
- `http://103.180.138.225:18000/docs`
- `http://103.180.138.225:18080`

Nếu dashboard mở được, đăng nhập bằng:

- Email: giá trị `ADMIN_EMAIL`
- Mật khẩu: giá trị `ADMIN_PASSWORD`

## 11. Cấu hình Desktop Tool để sync lên server

Trên máy CTSV, mở Desktop Tool và nhập:

- Backend API URL: `http://103.180.138.225:18000/api`
- API token: giá trị `DESKTOP_API_TOKEN` trong file `.env`

Sau đó:

1. bấm `Kiểm tra API`
2. chạy crawler hoặc `Chạy cào dữ liệu`
3. bấm `Đồng bộ ngay`

Khi sync thành công:

- backend trên server ghi dữ liệu vào PostgreSQL container
- backend tự gán label AI cho tin đã ingest
- frontend dashboard load dữ liệu mới qua API
- các chart và số liệu overview sẽ cập nhật theo dữ liệu đã ingest
- bảng tin sẽ hiện label cảm xúc:
  - `0`: tiêu cực
  - `1`: trung lập
  - `2`: tích cực

## 12. Trả lời câu hỏi quan trọng

### Desktop winform app có gửi data lên server Docker không?

Có.

Nếu Desktop Tool được cấu hình:

- `Backend API URL = http://103.180.138.225:18000/api`

thì sau khi crawl xong, tool sẽ:

1. đọc `posts.db` local trên máy CTSV
2. chuẩn hóa nội dung
3. tạo payload batch
4. gửi lên backend FastAPI đang chạy trong Docker trên server
5. backend ghi dữ liệu vào PostgreSQL trong Docker
6. frontend dashboard đọc lại dữ liệu đó qua API để hiển thị

### Desktop Tool có ghi trực tiếp vào database server không?

Không.

Desktop Tool chỉ gọi API backend.

Backend mới là thành phần ghi vào database PostgreSQL.

## 13. Lệnh vận hành thường dùng

Restart:

```bash
cd /opt/project02-2526/STEP6_Dashboard
docker compose restart
```

Build lại sau khi cập nhật code:

```bash
cd /opt/project02-2526/STEP6_Dashboard
docker compose up --build -d
```

Dừng hệ thống:

```bash
cd /opt/project02-2526/STEP6_Dashboard
docker compose down
```

Không xóa dữ liệu PostgreSQL volume:

- volume `ctsv_postgres_data` vẫn được giữ lại nếu chỉ chạy `docker compose down`

## 14. Gợi ý vận hành thực tế

- backend và frontend nên chạy ổn trên server trước
- sau đó mới cấu hình Desktop Tool trỏ lên server
- kiểm tra `Kiểm tra API` thành công rồi mới sync dữ liệu thật
- nên test 1 batch nhỏ trước khi chạy crawl diện rộng
