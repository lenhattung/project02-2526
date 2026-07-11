# Hướng dẫn sử dụng CTSV Desktop Scraper Tool

Tài liệu này hướng dẫn cán bộ CTSV chạy tool local để:

- lấy cookie Facebook
- chạy scraper
- đọc dữ liệu từ SQLite local
- đồng bộ dữ liệu lên Backend Dashboard

## 1. Nguyên tắc quan trọng

- Desktop Tool chạy trên máy local, không chạy scraper Facebook trên VPS
- Tool không lưu mật khẩu Facebook
- Scraper hiện tại cần file `cookies.json` hợp lệ trong `STEP1_ScrapingData`
- Nếu cookie hết hạn, cần lấy lại hoặc export lại `cookies.json`
- API token chỉ dùng để đồng bộ dữ liệu lên backend

## 2. Các file quan trọng

- `F:\project02-2526\STEP1_ScrapingData\scraper.py`
- `F:\project02-2526\STEP1_ScrapingData\requirements-scraper-core.txt`
- `F:\project02-2526\STEP1_ScrapingData\data\posts.db`
- `F:\project02-2526\STEP1_ScrapingData\cookies.json`

## 3. Chạy Desktop Tool

```powershell
cd F:\project02-2526\STEP1_ScrapingData\desktop_tool
.\.venv\Scripts\python.exe -m app.main
```

## 4. Cấu hình chuẩn local dev

Trong trang **Cấu hình**, kiểm tra:

- `Project folder`: `F:\project02-2526\STEP1_ScrapingData`
- `scraper.py`: `F:\project02-2526\STEP1_ScrapingData\scraper.py`
- `requirements.txt`: `F:\project02-2526\STEP1_ScrapingData\requirements-scraper-core.txt`
- `posts.db`: `F:\project02-2526\STEP1_ScrapingData\data\posts.db`
- `cookies.json`: `F:\project02-2526\STEP1_ScrapingData\cookies.json`
- `Backend API`: `http://localhost:8010/api`
- `API token`: `ctsv-demo-desktop-token`
- `Số bài mỗi batch`: `500`
- `Dừng khi trùng`: `10`

Lưu ý:

- App lưu config thật tại `%APPDATA%\CTSVScraperTool\config.json`
- Nếu trước đó đã lưu port cũ, cần sửa lại rồi bấm lưu

## 5. Chuẩn bị `cookies.json`

Scraper dùng `load_cookies(context)` và đọc trực tiếp file `cookies.json`, nên muốn crawl Facebook thì cần cookie.

Có 2 cách:

1. Bấm `Lấy cookie Facebook` trong Desktop Tool
2. Export thủ công cookie từ trình duyệt rồi đặt đúng file `cookies.json`

Yêu cầu:

- file JSON hợp lệ
- nội dung là danh sách cookie
- mỗi cookie tối thiểu có `name`, `value`, `domain`

## 6. Chạy cào dữ liệu

1. Mở trang **Cào dữ liệu**
2. Bấm `Kiểm tra cookie`
3. Nếu thiếu cookie, bấm `Lấy cookie Facebook`
4. Bấm `Chạy cào dữ liệu`
5. Theo dõi log ở trang **Logs**
6. Nếu cần dừng, bấm `Dừng`

Khi scraper chạy xong, dữ liệu local sẽ nằm trong:

- `F:\project02-2526\STEP1_ScrapingData\data\posts.db`

## 7. Đồng bộ dữ liệu lên backend

Điều kiện:

- backend đang chạy ở `http://localhost:8010`
- backend API URL trong app là `http://localhost:8010/api`
- API token đúng
- `posts.db` có dữ liệu

Thao tác:

1. Bấm `Kiểm tra API`
2. Vào trang **Đồng bộ**
3. Bấm `Đồng bộ ngay`

Tool sẽ:

- đọc SQLite local
- đưa nội dung qua bước ẩn danh từ `STEP2_Anonymize\Anonymize_CRF.py`
- chuẩn hóa nội dung sau ẩn danh
- tạo hash chống trùng
- gửi batch lên `/api/ingest/batches`

## 8. Nguồn dữ liệu chuẩn của hệ thống

- Scraper local ghi vào `STEP1_ScrapingData\data\posts.db`
- Desktop Tool đọc `posts.db` và sync lên backend
- Backend local dev lưu dữ liệu dashboard ở `STEP6_Dashboard\backend\ctsv_dashboard.db`
- Frontend dashboard chỉ đọc dữ liệu từ Backend API
- Backend sẽ tự gọi AI để gán nhãn cảm xúc sau khi ingest thành công

Flow chuẩn:

`Facebook -> scraper local -> posts.db -> Desktop Tool sync -> backend DB -> frontend dashboard`

Sau ingest, backend sẽ:

- gọi Gemini 1.5 Flash nếu có cấu hình API key
- fallback về mock nếu AI provider lỗi
- lưu nhãn cảm xúc:
  - `0`: tiêu cực
  - `1`: trung lập
  - `2`: tích cực

## 9. Đặt lịch chạy

Nhập một trong hai dạng:

```text
hours:4
daily:22:00
```

- `hours:4`: chạy mỗi 4 giờ
- `daily:22:00`: chạy mỗi ngày lúc 10 giờ tối

Nếu đóng cửa sổ app, tool sẽ ẩn xuống system tray và scheduler vẫn tiếp tục chạy.

## 10. Build file `.exe`

```powershell
cd F:\project02-2526\STEP1_ScrapingData\desktop_tool
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

Output:

- `F:\project02-2526\STEP1_ScrapingData\desktop_tool\dist\CTSVDesktopScraper`

## 11. Lỗi thường gặp

### Chưa có cookie

- Kiểm tra `cookies.json`
- Bấm `Lấy cookie Facebook`

### Backend offline

- Kiểm tra backend đã chạy ở port `8010`
- Kiểm tra đúng `Backend API URL`

### API token sai

- Backend trả `401` hoặc `403`
- Nhập lại token ở trang **Cấu hình**

### SQLite chưa có dữ liệu

- Kiểm tra `F:\project02-2526\STEP1_ScrapingData\data\posts.db`
- Chạy scraper trước khi sync
