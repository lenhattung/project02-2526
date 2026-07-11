# Hướng Dẫn Flow Desktop Tool Crawl Data Và Đẩy Lên VPS

## 1. Mục tiêu của Desktop Tool

Desktop Tool chạy trên máy cán bộ CTSV để:

- kiểm tra cookie Facebook
- chạy scraper local
- đọc SQLite local sau khi crawl
- ẩn danh dữ liệu bằng `STEP2_Anonymize/Anonymize_CRF.py`
- đồng bộ batch dữ liệu đã ẩn danh lên backend API trên VPS

Desktop Tool không ghi trực tiếp vào PostgreSQL trên server.

## 2. Kiến trúc dữ liệu chuẩn

Luồng dữ liệu chuẩn:

1. Facebook
2. `scraper.py`
3. SQLite local: `STEP1_ScrapingData/data/posts.db`
4. Desktop Tool
5. `STEP2_Anonymize/Anonymize_CRF.py`
6. Backend API trên VPS
7. PostgreSQL Docker trên VPS
8. Frontend Dashboard đọc dữ liệu từ backend

Điểm quan trọng:

- SQLite local chỉ là nơi lưu tạm dữ liệu crawl trên máy CTSV
- PostgreSQL trên VPS là nguồn dữ liệu online chính của dashboard

## 3. Cookie Facebook có bắt buộc không

Có.

Scraper hiện tại cần file:

```text
F:\project02-2526\STEP1_ScrapingData\cookies.json
```

Nếu thiếu hoặc cookie hết hạn:

- scraper không đăng nhập được Facebook
- không crawl được hoặc crawl thiếu dữ liệu
- Desktop Tool sẽ báo lỗi trạng thái cookie

Desktop Tool không lưu mật khẩu Facebook và không đăng nhập bằng tài khoản/mật khẩu.

## 4. Cấu hình cần điền trong Desktop Tool

Các đường dẫn chuẩn:

- `project_dir`: `F:\project02-2526\STEP1_ScrapingData`
- `scraper_path`: `F:\project02-2526\STEP1_ScrapingData\scraper.py`
- `requirements_path`: `F:\project02-2526\STEP1_ScrapingData\requirements-scraper-core.txt`
- `output_db_path`: `F:\project02-2526\STEP1_ScrapingData\data\posts.db`
- `cookies_path`: `F:\project02-2526\STEP1_ScrapingData\cookies.json`

Cấu hình backend VPS:

- `backend_url`: `http://103.180.138.225:18000/api`

Token:

- nhập đúng Desktop API token do backend cấp

## 5. Flow sử dụng hằng ngày

Thứ tự khuyến nghị:

1. Mở Desktop Tool
2. Kiểm tra trạng thái cookie
3. Kiểm tra API
4. Chạy crawl dữ liệu
5. Kiểm tra log crawl
6. Đồng bộ dữ liệu lên server
7. Kiểm tra dashboard web

Nếu bật `auto_sync_after_scrape`, sau khi crawl xong tool sẽ tự đồng bộ.

## 6. Bước crawl dữ liệu

Khi bấm nút chạy cào dữ liệu:

1. Tool gọi `scraper.py`
2. Scraper dùng Chrome/Playwright và cookie Facebook
3. Dữ liệu bài viết và bình luận được lưu vào:

```text
F:\project02-2526\STEP1_ScrapingData\data\posts.db
```

4. Tool đọc lại SQLite local
5. Tool map dữ liệu sang payload chuẩn để gửi backend

## 7. Bước ẩn danh trước khi đẩy server

Trước khi gửi lên VPS, Desktop Tool sẽ chạy ẩn danh qua:

```text
F:\project02-2526\STEP2_Anonymize\Anonymize_CRF.py
```

Nghĩa là:

- dữ liệu đẩy lên backend là dữ liệu đã đi qua bước ẩn danh
- dashboard không dùng bản thô local để hiển thị

Nếu pipeline ẩn danh lỗi:

- tool sẽ fallback giữ nguyên nội dung để không làm gãy sync
- cần kiểm tra lại môi trường ẩn danh nếu muốn bảo đảm tuyệt đối

## 8. Bước sync lên VPS

Desktop Tool gửi dữ liệu qua endpoint:

```text
POST http://103.180.138.225:18000/api/ingest/batches
```

Header dùng:

```text
X-API-Token: <desktop_token>
```

Payload gồm:

- thông tin nguồn crawl
- danh sách bài viết
- danh sách comment
- `content_hash` để tránh trùng
- metadata cho biết dữ liệu đã qua anonymize

## 9. Sau khi sync thành công thì chuyện gì xảy ra

Backend trên VPS sẽ:

1. nhận batch từ Desktop Tool
2. upsert vào PostgreSQL
3. chạy label AI ở background
4. ghi các cột:
   - `gemini_label`
   - `simcse_label`
   - `phobert_label`
   - `bgem3_label`
   - `voted_label`
5. frontend dashboard đọc dữ liệu mới để hiển thị

## 10. Đặt lịch tự động cào

Desktop Tool có thể cấu hình schedule local, ví dụ:

- mỗi ngày lúc `22:00`
- hoặc mỗi `X` giờ

Nếu app đang chạy nền:

- tới đúng giờ sẽ tự chạy crawl
- sau đó tự sync nếu đã bật `auto_sync_after_scrape`

Điều này phù hợp với mô hình:

- máy CTSV là nơi crawl
- server VPS là nơi lưu và hiển thị

## 11. Chạy nền và icon khay hệ thống

Khi build `.exe`, Desktop Tool có thể chạy nền ở system tray:

- đóng cửa sổ chính không đồng nghĩa tắt hẳn app
- app vẫn có thể tiếp tục giữ lịch chạy
- chỉ khi chọn `Quit` hoặc `Close` từ tray icon thì mới dừng hẳn

Vì vậy nếu muốn auto crawl theo lịch, cần để app chạy nền.

## 12. Cách kiểm tra đã đẩy dữ liệu lên server chưa

Kiểm tra lần lượt:

1. Desktop Tool báo sync thành công
2. Backend API `http://103.180.138.225:18000/health` còn sống
3. Dashboard `http://103.180.138.225:18080` login được
4. Trang quản lý tin có record mới
5. Record có cột label hoặc trạng thái label

## 13. Các lỗi thường gặp

`Không có cookie`:

- đặt lại `cookies.json` đúng vị trí
- kiểm tra cookie còn hạn

`Test API failed`:

- backend VPS chưa chạy
- sai `backend_url`
- token sai

`Sync 401/403`:

- `X-API-Token` không hợp lệ
- backend đã đổi token

`Dashboard không có dữ liệu mới`:

- crawl chưa ghi được vào SQLite
- sync chưa chạy
- backend ingest lỗi

`Có dữ liệu nhưng chưa có nhãn local model`:

- model trên VPS chưa mount đúng
- backend thiếu dependency hoặc thiếu RAM

## 14. Kết luận

Flow đúng và chuẩn hóa là:

- crawl ở local bằng Desktop Tool
- lưu tạm vào SQLite local
- ẩn danh trước khi sync
- sync qua Backend API
- lưu PostgreSQL trên VPS
- backend gán nhãn AI
- frontend dashboard hiển thị dữ liệu và chart từ backend

Frontend dashboard không đọc trực tiếp dữ liệu từ `STEP1_ScrapingData\data`.
