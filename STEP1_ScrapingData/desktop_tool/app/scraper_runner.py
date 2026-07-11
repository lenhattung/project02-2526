from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

from app.config import AppConfig


COOKIE_CAPTURE_INLINE_SCRIPT = r"""
from pathlib import Path
import json
import sys
import time
from playwright.sync_api import sync_playwright

def has_facebook_session(cookies):
    names = {cookie.get("name") for cookie in cookies if "facebook.com" in cookie.get("domain", "")}
    return "c_user" in names and "xs" in names

output_path = Path(sys.argv[1]).resolve()
timeout_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 600
output_path.parent.mkdir(parents=True, exist_ok=True)
print("COOKIE_INFO:Opening Facebook browser for cookie capture.")
print("COOKIE_INFO:Please sign in to Facebook in the opened browser if needed.")

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.facebook.com/", wait_until="domcontentloaded")
    deadline = time.time() + timeout_seconds
    status_tick = 0

    while time.time() < deadline:
        page.wait_for_timeout(2000)
        all_cookies = context.cookies()
        if has_facebook_session(all_cookies):
            output_path.write_text(json.dumps(all_cookies, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"COOKIE_CAPTURED:{output_path}")
            browser.close()
            raise SystemExit(0)

        status_tick += 1
        if status_tick % 5 == 0:
            if "login" in page.url.lower():
                print("COOKIE_INFO:Waiting for Facebook sign-in...")
            else:
                print("COOKIE_INFO:Waiting for Facebook session cookies...")

    print("COOKIE_TIMEOUT:Timed out before session cookies were captured.")
    browser.close()
    raise SystemExit(2)
"""

COOKIE_INFO_TRANSLATIONS = {
    "Opening Facebook browser for cookie capture.": "Sắp mở trình duyệt Facebook để lấy cookie.",
    "Please sign in to Facebook in the opened browser if needed.": "Nếu Facebook yêu cầu, hãy đăng nhập trong cửa sổ trình duyệt vừa mở.",
    "Waiting for Facebook sign-in...": "Đang chờ bạn đăng nhập Facebook...",
    "Waiting for Facebook session cookies...": "Đang chờ Facebook cấp đủ cookie phiên đăng nhập...",
}


@dataclass(frozen=True)
class CookieStatus:
    code: str
    title: str
    message: str
    cookie_count: int = 0
    ok: bool = False


def check_cookie_file(path: str) -> CookieStatus:
    cookie_path = Path(path)
    if not cookie_path.exists():
        return CookieStatus(
            code="missing",
            title="Chưa có cookie",
            message=f"Không tìm thấy cookies.json tại: {cookie_path}",
        )
    try:
        data = json.loads(cookie_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CookieStatus(
            code="invalid_json",
            title="Cookie sai định dạng",
            message=f"File cookies.json không phải JSON hợp lệ: {exc}",
        )
    except OSError as exc:
        return CookieStatus(
            code="read_error",
            title="Không đọc được cookie",
            message=str(exc),
        )
    if not isinstance(data, list) or not data:
        return CookieStatus(
            code="invalid_shape",
            title="Cookie sai định dạng",
            message="cookies.json phải là danh sách cookie export từ trình duyệt.",
        )
    required_keys = {"name", "value", "domain"}
    if not all(isinstance(item, dict) and required_keys.issubset(item.keys()) for item in data):
        return CookieStatus(
            code="invalid_cookie",
            title="Cookie thiếu thông tin",
            message="Mỗi cookie cần có tối thiểu name, value và domain.",
        )
    return CookieStatus(
        code="ok",
        title="Đọc được cookie",
        message=f"Đã đọc được {len(data)} cookie. Vẫn cần chạy thử để biết session Facebook còn hạn hay không.",
        cookie_count=len(data),
        ok=True,
    )


def venv_python(project_dir: str) -> Path:
    base = Path(project_dir) / ".venv"
    if os.name == "nt":
        return base / "Scripts" / "python.exe"
    return base / "bin" / "python"


def _stream_process_output(process: subprocess.Popen, emit) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        emit(line.rstrip())


def setup_environment(config: AppConfig, emit) -> None:
    project_dir = Path(config.project_dir)
    env_dir = project_dir / ".venv"
    if not env_dir.exists():
        emit(f"Creating virtual environment: {env_dir}")
        venv.create(env_dir, with_pip=True)

    python_path = venv_python(config.project_dir)
    emit(f"Using Python: {python_path}")

    req = Path(config.requirements_path)
    core_req = project_dir / "requirements-scraper-core.txt"
    install_req = core_req if core_req.exists() else req
    if install_req.exists():
        emit(f"Installing scraper requirements: {install_req}")
        process = subprocess.Popen(
            [str(python_path), "-m", "pip", "install", "-r", str(install_req)],
            cwd=config.project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        _stream_process_output(process, emit)
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"pip install failed with code {code}")

        emit("Ensuring Playwright Chromium is installed...")
        browser_install = subprocess.Popen(
            [str(python_path), "-m", "playwright", "install", "chromium"],
            cwd=config.project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        _stream_process_output(browser_install, emit)
        browser_code = browser_install.wait()
        if browser_code != 0:
            emit(f"Warning: playwright install chromium returned code {browser_code}.")
    else:
        emit("Scraper requirements file not found; skipping install.")


def capture_facebook_cookies(config: AppConfig, emit, timeout_seconds: int = 600) -> Path:
    python_path = venv_python(config.project_dir)
    if not python_path.exists():
        raise RuntimeError("Chưa tìm thấy .venv của scraper. Hãy bấm 'Cài môi trường' trước.")

    worker_script = Path(__file__).with_name("cookie_capture_worker.py")
    emit(f"Chuẩn bị lấy cookie và lưu tại: {config.cookies_path}")
    command = [str(python_path), str(worker_script), str(config.cookies_path), str(timeout_seconds)]
    if not worker_script.exists():
        command = [str(python_path), "-c", COOKIE_CAPTURE_INLINE_SCRIPT, str(config.cookies_path), str(timeout_seconds)]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=config.project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert process.stdout is not None
    saved_path: Path | None = None
    did_timeout = False
    last_error_message: str | None = None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if line.startswith("COOKIE_CAPTURED:"):
            saved_path = Path(line.split(":", 1)[1].strip())
            emit(f"Đã lưu cookie vào: {saved_path}")
        elif line.startswith("COOKIE_TIMEOUT:"):
            did_timeout = True
            emit("Hết thời gian chờ đăng nhập Facebook hoặc chưa đủ cookie phiên.")
        elif line.startswith("COOKIE_INFO:"):
            info = line.split(":", 1)[1].strip()
            emit(COOKIE_INFO_TRANSLATIONS.get(info, info))
        elif line.startswith("COOKIE_ERROR:"):
            last_error_message = line.split(":", 1)[1].strip()
            emit(last_error_message)
        else:
            emit(line)

    code = process.wait()
    if code != 0:
        if did_timeout:
            raise RuntimeError("Chưa lấy được cookie vì quá thời gian chờ hoặc bạn chưa đăng nhập Facebook xong.")
        if last_error_message:
            raise RuntimeError(f"Không lấy được cookie Facebook: {last_error_message}")
        raise RuntimeError("Không lấy được cookie Facebook. Hãy kiểm tra log và đảm bảo Playwright Chromium đã sẵn sàng.")
    if not saved_path:
        raise RuntimeError("Trình duyệt đã đóng nhưng chưa lưu được cookie Facebook.")
    return saved_path


class ScraperProcess:
    def __init__(self, config: AppConfig, emit):
        self.config = config
        self.emit = emit
        self.process: subprocess.Popen | None = None

    def run(self) -> int:
        cookie_status = check_cookie_file(self.config.cookies_path)
        if not cookie_status.ok:
            self.emit(f"{cookie_status.title}: {cookie_status.message}")
            raise RuntimeError("Không thể chạy scraper vì cookies.json chưa hợp lệ.")

        python_path = venv_python(self.config.project_dir)
        if not python_path.exists():
            python_path = Path(sys.executable)

        env = os.environ.copy()
        env["CTSV_SCRAPER_AUTO_CLOSE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["CTSV_MAX_POSTS_PER_PAGE"] = str(self.config.max_posts_per_page)
        env["CTSV_DUPLICATE_STOP_THRESHOLD"] = str(self.config.duplicate_stop_threshold)
        self.emit(f"Cookie OK: {cookie_status.cookie_count} cookie. Bắt đầu chạy scraper.")
        self.emit(f"Running scraper: {python_path} {self.config.scraper_path}")
        self.process = subprocess.Popen(
            [str(python_path), str(self.config.scraper_path)],
            cwd=self.config.project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        _stream_process_output(self.process, self.emit)
        return self.process.wait(timeout=self.config.timeout_seconds)

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.emit("Stopping scraper process...")
            self.process.terminate()
