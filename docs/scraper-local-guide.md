# Scraper Local Guide

Facebook scraper phải chạy trên máy CTSV vì phụ thuộc:

- trạng thái đăng nhập Facebook
- cookie phiên local
- Playwright Chromium
- môi trường mạng tại máy cán bộ

## Run thủ công

```powershell
cd F:\project02-2526\STEP1_ScrapingData
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-scraper-core.txt
.\.venv\Scripts\python.exe scraper.py
```

## Output scraper

Scraper hiện ghi vào:

- `F:\project02-2526\STEP1_ScrapingData\data\posts.db`

Các bảng chính:

- `posts`
- `comments`

## Quan hệ với Desktop Tool và Dashboard

- Desktop Tool đọc `posts.db`
- Desktop Tool chuẩn hóa dữ liệu và sync lên Backend API
- Backend lưu dữ liệu chuẩn cho dashboard
- Frontend dashboard không đọc trực tiếp `posts.db`

## Ghi chú vận hành

- Giữ Facebook đang đăng nhập trên máy CTSV
- Làm mới `cookies.json` khi scraper báo login failure
- Không lưu tài khoản Facebook trong source code
- Nên backup `posts.db` trước các đợt crawl lớn
