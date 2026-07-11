# Desktop EXE Build Guide

## Build

```powershell
cd F:\project02-2526\STEP1_ScrapingData\desktop_tool
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

Output:

- `F:\project02-2526\STEP1_ScrapingData\desktop_tool\dist\CTSVDesktopScraper`

## Cấu hình cần kiểm tra trước khi bàn giao

- Backend API URL: `http://localhost:8010/api` cho local dev
- API token: `ctsv-demo-desktop-token` hoặc token tạo từ dashboard
- SQLite output: `F:\project02-2526\STEP1_ScrapingData\data\posts.db`
- Cookies path: `F:\project02-2526\STEP1_ScrapingData\cookies.json`

## Smoke test sau build

1. Mở file `.exe`
2. Kiểm tra app mở được và xuống system tray bình thường
3. Bấm `Kiểm tra API`
4. Bấm `Đồng bộ ngay`
5. Kiểm tra dashboard đã thấy dữ liệu mới
