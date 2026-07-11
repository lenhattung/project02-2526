from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import traceback

from apscheduler.schedulers.background import BackgroundScheduler
from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QSystemTrayIcon,
    QMenu,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig, get_api_token, load_config, parse_schedule, save_config, set_api_token
from app.scraper_runner import CookieStatus, ScraperProcess, capture_facebook_cookies, check_cookie_file, setup_environment
from app.sqlite_adapter import build_ingest_batch, get_local_counts
from app.sync_client import SyncClient


class Worker(QObject):
    log = Signal(str)
    done = Signal(bool, str)
    metrics = Signal(int, int)

    def __init__(self, task):
        super().__init__()
        self.task = task

    def run(self):
        try:
            message = self.task(self.log.emit, self.metrics.emit)
            self.done.emit(True, message or "Hoàn tất")
        except Exception as exc:
            self.log.emit(traceback.format_exc())
            self.done.emit(False, str(exc))


class StatusCard(QFrame):
    def __init__(self, title: str, value: str = "-", hint: str = ""):
        super().__init__()
        self.setObjectName("StatusCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(148)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("CardValue")
        self.hint_label = QLabel(hint)
        self.hint_label.setObjectName("CardHint")
        self.hint_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.hint_label)

    def set(self, value: str, hint: str | None = None) -> None:
        self.value_label.setText(value)
        if hint is not None:
            self.hint_label.setText(hint)


def build_app_icon(size: int = 64) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    outer_rect = pixmap.rect().adjusted(4, 4, -4, -4)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#0d5c57"))
    painter.drawRoundedRect(outer_rect, 14, 14)

    inner_rect = outer_rect.adjusted(8, 8, -8, -8)
    painter.setBrush(QColor("#dff2e7"))
    painter.drawRoundedRect(inner_rect, 10, 10)

    painter.setPen(QPen(QColor("#0d5c57")))
    font = QFont("Segoe UI", max(12, size // 4))
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(inner_rect, Qt.AlignmentFlag.AlignCenter, "CT")

    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    schedule_run_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CTSV Scraper Control Center")
        self.resize(1280, 820)
        self.setWindowIcon(build_app_icon())
        self.config = load_config()
        self.scraper: ScraperProcess | None = None
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        self.active_thread: QThread | None = None
        self.active_worker: Worker | None = None
        self.allow_quit = False
        self.tray_icon: QSystemTrayIcon | None = None
        self.log_entries: list[str] = []
        self.last_error = "Chưa có lỗi"
        self.total_synced = 0
        self.cookie_status: CookieStatus = check_cookie_file(self.config.cookies_path)
        self._build_ui()
        self._build_tray()
        self.schedule_run_requested.connect(self.run_scraper)
        self._load_fields()
        self.refresh_local_state()
        self.refresh_cookie_status()
        self.restore_schedule_if_enabled()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 22, 20, 20)
        sidebar_layout.setSpacing(14)

        brand_panel = QFrame()
        brand_panel.setObjectName("BrandPanel")
        brand_layout = QVBoxLayout(brand_panel)
        brand_layout.setContentsMargins(16, 16, 16, 16)
        brand_layout.setSpacing(6)
        brand = QLabel("CTSV\nScraper")
        brand.setObjectName("Brand")
        brand_desc = QLabel("Điều khiển quy trình cào dữ liệu và đồng bộ backend trên máy CTSV.")
        brand_desc.setObjectName("BrandDesc")
        brand_desc.setWordWrap(True)
        brand_layout.addWidget(brand)
        brand_layout.addWidget(brand_desc)
        sidebar_layout.addWidget(brand_panel)

        self.sidebar_status = QLabel("Sẵn sàng")
        self.sidebar_status.setObjectName("SidebarPill")
        sidebar_layout.addWidget(self.sidebar_status)

        self.sidebar_cookie = QLabel("Cookie: đang kiểm tra")
        self.sidebar_cookie.setObjectName("SidebarSubPill")
        sidebar_layout.addWidget(self.sidebar_cookie)

        self.sidebar = QListWidget()
        self.sidebar.setObjectName("Navigation")
        self.sidebar.setSpacing(6)
        for label in ["Tổng quan", "Cào dữ liệu", "Đồng bộ", "Lịch chạy", "Cấu hình", "Logs"]:
            item = QListWidgetItem(label)
            self.sidebar.addItem(item)
        self.sidebar.currentRowChanged.connect(self._switch_page)
        sidebar_layout.addWidget(self.sidebar, 1)

        footer = QLabel("Chạy local trên máy CTSV\nKhông lưu mật khẩu Facebook")
        footer.setObjectName("SidebarFooter")
        footer.setWordWrap(True)
        sidebar_layout.addWidget(footer)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(30, 24, 30, 24)
        content_layout.setSpacing(20)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 22, 24, 22)
        header_layout.setSpacing(14)

        header_text = QVBoxLayout()
        header_text.setSpacing(6)
        title = QLabel("CTSV Scraper Control Center")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel("Kiểm tra cookie, lấy cookie Facebook, cào dữ liệu local và đồng bộ lên Dashboard.")
        subtitle.setObjectName("HeaderSubtitle")
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header_layout.addLayout(header_text, 1)

        right_group = QVBoxLayout()
        right_group.setSpacing(10)
        right_group.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.system_status = QLabel("Idle")
        self.system_status.setObjectName("StatusChip")
        right_group.addWidget(self.system_status, 0, Qt.AlignmentFlag.AlignRight)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        quick_capture = QPushButton("Lấy cookie")
        quick_capture.setObjectName("GhostButton")
        quick_capture.clicked.connect(self.capture_cookies)
        action_row.addWidget(quick_capture)
        quick_run = QPushButton("Chạy cào dữ liệu")
        quick_run.setObjectName("PrimaryButton")
        quick_run.clicked.connect(self.run_scraper)
        action_row.addWidget(quick_run)
        quick_resume = QPushButton("Tiếp tục cào")
        quick_resume.setObjectName("GhostButton")
        quick_resume.clicked.connect(self.resume_scraper)
        action_row.addWidget(quick_resume)
        quick_sync = QPushButton("Đồng bộ ngay")
        quick_sync.setObjectName("GhostButton")
        quick_sync.clicked.connect(self.sync_now)
        action_row.addWidget(quick_sync)
        right_group.addLayout(action_row)
        header_layout.addLayout(right_group)
        content_layout.addWidget(header)

        self.pages = QStackedWidget()
        self.pages.setObjectName("PageStack")
        content_layout.addWidget(self.pages, 1)

        shell.addWidget(sidebar)
        shell.addWidget(content, 1)
        self.setCentralWidget(root)

        self._build_overview_page()
        self._build_crawl_page()
        self._build_sync_page()
        self._build_schedule_page()
        self._build_config_page()
        self._build_logs_page()
        self.sidebar.setCurrentRow(0)

    def _scroll_page(self) -> tuple[QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("PageScroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.viewport().setObjectName("PageViewport")
        page = QWidget()
        page.setObjectName("PageBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(20)
        scroll.setWidget(page)
        self.pages.addWidget(scroll)
        return page, layout

    def _page_heading(self, title: str, subtitle: str) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        heading = QLabel(title)
        heading.setObjectName("PageTitle")
        text = QLabel(subtitle)
        text.setObjectName("PageSubtitle")
        text.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(text)
        return box

    def _panel(self, title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)
        heading = QLabel(title)
        heading.setObjectName("PanelTitle")
        layout.addWidget(heading)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("PanelSubtitle")
            sub.setWordWrap(True)
            layout.addWidget(sub)
        return panel, layout

    def _button(self, text: str, handler, kind: str = "GhostButton") -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(kind)
        button.clicked.connect(handler)
        return button

    def _field_row(self, label: str, widget: QWidget, browse_handler=None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        name = QLabel(label)
        name.setObjectName("FieldLabel")
        name.setFixedWidth(160)
        row.addWidget(name)
        row.addWidget(widget, 1)
        if browse_handler:
            row.addWidget(self._button("Chọn", browse_handler, "SmallButton"))
        return row

    def _build_overview_page(self) -> None:
        _, layout = self._scroll_page()
        layout.addWidget(self._page_heading("Tổng quan vận hành", "Theo dõi trạng thái cào dữ liệu, cookie, dữ liệu local và đồng bộ backend."))

        start_panel, start_layout = self._panel(
            "Bắt đầu nhanh",
            "Luồng dùng dễ nhất: lấy hoặc kiểm tra cookie, kiểm tra API, chạy cào dữ liệu, rồi đồng bộ. Nếu bật tự đồng bộ thì chỉ cần chạy scraper.",
        )
        start_row = QHBoxLayout()
        start_row.setSpacing(12)
        for label in ["1. Lấy hoặc kiểm tra cookie", "2. Kiểm tra API", "3. Chạy cào dữ liệu", "4. Theo dõi kết quả"]:
            chip = QLabel(label)
            chip.setObjectName("GuideChip")
            start_row.addWidget(chip)
        start_layout.addLayout(start_row)
        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        action_row.addWidget(self._button("Lấy cookie Facebook", self.capture_cookies, "GhostButton"))
        action_row.addWidget(self._button("Kiểm tra cookie", self.refresh_cookie_status, "GhostButton"))
        action_row.addWidget(self._button("Kiểm tra API", self.test_api, "GhostButton"))
        action_row.addWidget(self._button("Cài môi trường", self.setup_env, "GhostButton"))
        action_row.addWidget(self._button("Chạy cào dữ liệu", self.run_scraper, "PrimaryButton"))
        action_row.addWidget(self._button("Tiếp tục cào", self.resume_scraper, "GhostButton"))
        start_layout.addLayout(action_row)
        layout.addWidget(start_panel)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        self.status_card = StatusCard("Trạng thái", "Idle", "Chưa có tác vụ đang chạy")
        self.cookie_card = StatusCard("Cookie Facebook", "-", "Cần cookies.json hợp lệ")
        self.local_card = StatusCard("Dữ liệu local", "0 bài", "0 bình luận trong SQLite")
        self.synced_card = StatusCard("Đã đồng bộ", "0", "Tính theo phiên chạy hiện tại")
        self.last_run_card = StatusCard("Lần chạy gần nhất", "Chưa chạy", "Scraper chưa được gọi")
        self.error_card = StatusCard("Lỗi gần nhất", "Chưa có lỗi", "Lỗi sẽ hiển thị rõ tại đây")
        for index, card in enumerate([self.status_card, self.cookie_card, self.local_card, self.synced_card, self.last_run_card, self.error_card]):
            grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(grid)

        panel, panel_layout = self._panel("Thông tin cần chú ý")
        self.overview_notice = QLabel("Nếu đang báo thiếu hoặc sai cookie, hãy vào trang Cào dữ liệu và bấm 'Lấy cookie Facebook' trước khi chạy.")
        self.overview_notice.setObjectName("PanelSubtitle")
        self.overview_notice.setWordWrap(True)
        panel_layout.addWidget(self.overview_notice)
        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self._button("Mở trang Cào dữ liệu", lambda: self.sidebar.setCurrentRow(1), "GhostButton"))
        actions.addWidget(self._button("Đồng bộ ngay", self.sync_now, "GhostButton"))
        panel_layout.addLayout(actions)
        layout.addWidget(panel)
        layout.addStretch()

    def _build_crawl_page(self) -> None:
        _, layout = self._scroll_page()
        layout.addWidget(self._page_heading("Cào dữ liệu Facebook", "Chạy scraper.py bằng môi trường local, lấy cookie bằng trình duyệt và hỗ trợ tự đồng bộ sau khi cào."))

        guide, guide_layout = self._panel(
            "Trước khi cào dữ liệu",
            "Scraper hiện tại cần session Facebook hợp lệ từ cookies.json. Tool không lấy mật khẩu và không hardcode tài khoản.",
        )
        for text in [
            "1. Bấm 'Lấy cookie Facebook' nếu muốn app tự mở trình duyệt và tự lưu cookies.json.",
            "2. Nếu không lấy tự động, bạn vẫn có thể export cookies.json thủ công từ trình duyệt đã đăng nhập Facebook.",
            "3. Khi chạy, Chrome hoặc Playwright có thể mở trình duyệt để truy cập bài viết.",
            "4. Nếu Facebook yêu cầu đăng nhập lại, hãy lấy lại cookie rồi chạy lại.",
        ]:
            label = QLabel(text)
            label.setObjectName("GuideLine")
            guide_layout.addWidget(label)
        layout.addWidget(guide)

        cookie_panel, cookie_layout = self._panel("Trạng thái cookie")
        self.cookie_title = QLabel("-")
        self.cookie_title.setObjectName("CookieTitle")
        self.cookie_message = QLabel("-")
        self.cookie_message.setObjectName("PanelSubtitle")
        self.cookie_message.setWordWrap(True)
        cookie_layout.addWidget(self.cookie_title)
        cookie_layout.addWidget(self.cookie_message)
        cookie_actions = QHBoxLayout()
        cookie_actions.setSpacing(10)
        cookie_actions.addWidget(self._button("Lấy cookie Facebook", self.capture_cookies, "PrimaryButton"))
        cookie_actions.addWidget(self._button("Kiểm tra cookie", self.refresh_cookie_status, "GhostButton"))
        cookie_actions.addWidget(self._button("Chọn cookies.json", self.select_cookie_file, "GhostButton"))
        cookie_actions.addWidget(self._button("Mở thư mục scraper", self.open_project_folder, "GhostButton"))
        cookie_layout.addLayout(cookie_actions)
        layout.addWidget(cookie_panel)

        run_panel, run_layout = self._panel("Hành động crawl")
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self._button("Cài môi trường", self.setup_env, "GhostButton"))
        row.addWidget(self._button("Chạy cào dữ liệu", self.run_scraper, "PrimaryButton"))
        row.addWidget(self._button("Tiếp tục cào", self.resume_scraper, "GhostButton"))
        row.addWidget(self._button("Dừng", self.stop_scraper, "DangerButton"))
        row.addWidget(self._button("Đồng bộ ngay", self.sync_now, "GhostButton"))
        run_layout.addLayout(row)
        self.auto_sync_hint = QLabel("Tự đồng bộ sau khi cào: bật")
        self.auto_sync_hint.setObjectName("PanelSubtitle")
        run_layout.addWidget(self.auto_sync_hint)
        layout.addWidget(run_panel)
        layout.addStretch()

    def _build_sync_page(self) -> None:
        _, layout = self._scroll_page()
        layout.addWidget(self._page_heading("Đồng bộ dữ liệu", "Đọc SQLite local, chuẩn hóa nội dung, tạo hash chống trùng và gửi batch lên Backend API."))
        panel, panel_layout = self._panel("Trạng thái sync")
        self.sync_summary = QLabel("Chưa đồng bộ trong phiên này.")
        self.sync_summary.setObjectName("PanelSubtitle")
        panel_layout.addWidget(self.sync_summary)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self._button("Refresh số liệu local", self.refresh_local_state, "GhostButton"))
        row.addWidget(self._button("Kiểm tra API", self.test_api, "GhostButton"))
        row.addWidget(self._button("Đồng bộ ngay", self.sync_now, "PrimaryButton"))
        panel_layout.addLayout(row)
        layout.addWidget(panel)
        layout.addStretch()

    def _build_schedule_page(self) -> None:
        _, layout = self._scroll_page()
        layout.addWidget(self._page_heading("Lịch chạy tự động", "Scheduler chạy khi Desktop Tool đang mở. Dùng hours:X hoặc daily:HH:MM."))
        panel, panel_layout = self._panel("Cấu hình lịch")
        self.schedule_value = QLineEdit()
        self.schedule_value.setPlaceholderText("daily:22:00 hoặc hours:4")
        panel_layout.addLayout(self._field_row("Chu kỳ", self.schedule_value))
        self.schedule_status = QLabel("Chưa bật lịch chạy")
        self.schedule_status.setObjectName("PanelSubtitle")
        panel_layout.addWidget(self.schedule_status)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self._button("Bật lịch chạy", self.enable_schedule, "PrimaryButton"))
        row.addWidget(self._button("Tắt lịch chạy", self.disable_schedule, "GhostButton"))
        panel_layout.addLayout(row)
        layout.addWidget(panel)
        layout.addStretch()

    def _build_config_page(self) -> None:
        _, layout = self._scroll_page()
        layout.addWidget(self._page_heading("Cấu hình", "Lưu tại %APPDATA%\\CTSVScraperTool. API token lưu qua keyring nếu máy hỗ trợ."))

        panel, panel_layout = self._panel("Đường dẫn local")
        self.project_dir = QLineEdit()
        self.scraper_path = QLineEdit()
        self.requirements_path = QLineEdit()
        self.output_db_path = QLineEdit()
        self.cookies_path = QLineEdit()
        panel_layout.addLayout(self._field_row("Project folder", self.project_dir, lambda: self.browse_folder(self.project_dir)))
        panel_layout.addLayout(self._field_row("scraper.py", self.scraper_path, lambda: self.browse_file(self.scraper_path, "Python files (*.py)")))
        panel_layout.addLayout(self._field_row("requirements.txt", self.requirements_path, lambda: self.browse_file(self.requirements_path, "Text files (*.txt)")))
        panel_layout.addLayout(self._field_row("posts.db", self.output_db_path, lambda: self.browse_file(self.output_db_path, "SQLite DB (*.db);;All files (*)")))
        panel_layout.addLayout(self._field_row("cookies.json", self.cookies_path, lambda: self.browse_file(self.cookies_path, "JSON files (*.json)")))
        layout.addWidget(panel)

        api_panel, api_layout = self._panel("Backend và đồng bộ")
        self.backend_url = QLineEdit()
        self.api_token = QLineEdit()
        self.api_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.timeout_seconds = QSpinBox()
        self.timeout_seconds.setRange(60, 86400)
        self.timeout_seconds.setSuffix(" giây")
        self.sync_limit = QSpinBox()
        self.sync_limit.setRange(1, 5000)
        self.max_posts_per_page = QSpinBox()
        self.max_posts_per_page.setRange(1, 10000)
        self.duplicate_stop_threshold = QSpinBox()
        self.duplicate_stop_threshold.setRange(1, 200)
        self.auto_sync_after_scrape = QCheckBox("Tự đồng bộ sau khi cào xong")
        api_layout.addLayout(self._field_row("Backend API", self.backend_url))
        api_layout.addLayout(self._field_row("API token", self.api_token))
        api_layout.addLayout(self._field_row("Timeout scraper", self.timeout_seconds))
        api_layout.addLayout(self._field_row("Số bài mỗi batch", self.sync_limit))
        api_layout.addLayout(self._field_row("Giới hạn an toàn", self.max_posts_per_page))
        api_layout.addLayout(self._field_row("Dừng khi trùng", self.duplicate_stop_threshold))
        api_layout.addWidget(self.auto_sync_after_scrape)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self._button("Lưu cấu hình", self.save_fields, "PrimaryButton"))
        row.addWidget(self._button("Kiểm tra API", self.test_api, "GhostButton"))
        row.addWidget(self._button("Kiểm tra cookie", self.refresh_cookie_status, "GhostButton"))
        api_layout.addLayout(row)
        layout.addWidget(api_panel)
        layout.addStretch()

    def _build_logs_page(self) -> None:
        _, layout = self._scroll_page()
        layout.addWidget(self._page_heading("Logs", "Theo dõi realtime quá trình setup, lấy cookie, crawl và sync."))
        panel, panel_layout = self._panel("Nhật ký")
        filter_row = QHBoxLayout()
        self.log_filter = QComboBox()
        self.log_filter.addItems(["Tất cả", "Info", "Warning", "Error"])
        self.log_filter.currentTextChanged.connect(self.render_logs)
        filter_row.addWidget(self.log_filter)
        filter_row.addStretch()
        filter_row.addWidget(self._button("Copy log", self.copy_logs, "GhostButton"))
        filter_row.addWidget(self._button("Export log", self.export_logs, "GhostButton"))
        panel_layout.addLayout(filter_row)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(430)
        panel_layout.addWidget(self.log_box)
        layout.addWidget(panel)

    def _switch_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)

    def _load_fields(self) -> None:
        self.project_dir.setText(self.config.project_dir)
        self.scraper_path.setText(self.config.scraper_path)
        self.requirements_path.setText(self.config.requirements_path)
        self.output_db_path.setText(self.config.output_db_path)
        self.cookies_path.setText(self.config.cookies_path)
        self.backend_url.setText(self.config.backend_url)
        self.timeout_seconds.setValue(self.config.timeout_seconds)
        self.sync_limit.setValue(self.config.sync_limit)
        self.max_posts_per_page.setValue(self.config.max_posts_per_page)
        self.duplicate_stop_threshold.setValue(self.config.duplicate_stop_threshold)
        self.auto_sync_after_scrape.setChecked(self.config.auto_sync_after_scrape)
        self.schedule_value.setText(f"{self.config.schedule_type}:{self.config.schedule_value}")
        self.api_token.setText(get_api_token())
        self.auto_sync_hint.setText(f"Tự đồng bộ sau khi cào: {'bật' if self.config.auto_sync_after_scrape else 'tắt'}")

    def save_fields(self) -> None:
        schedule_type, schedule_value = parse_schedule(self.schedule_value.text())
        self.config = AppConfig(
            project_dir=self.project_dir.text().strip(),
            scraper_path=self.scraper_path.text().strip(),
            requirements_path=self.requirements_path.text().strip(),
            output_db_path=self.output_db_path.text().strip(),
            cookies_path=self.cookies_path.text().strip(),
            backend_url=self.backend_url.text().strip().rstrip("/"),
            source_name=self.config.source_name,
            source_url=self.config.source_url,
            timeout_seconds=self.timeout_seconds.value(),
            max_posts_per_page=self.max_posts_per_page.value(),
            duplicate_stop_threshold=self.duplicate_stop_threshold.value(),
            auto_sync_after_scrape=self.auto_sync_after_scrape.isChecked(),
            sync_limit=self.sync_limit.value(),
            schedule_enabled=self.config.schedule_enabled,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
        )
        save_config(self.config)
        set_api_token(self.api_token.text().strip())
        self.auto_sync_hint.setText(f"Tự đồng bộ sau khi cào: {'bật' if self.config.auto_sync_after_scrape else 'tắt'}")
        self.log("INFO Đã lưu cấu hình local.")
        self.refresh_cookie_status()
        self.refresh_local_state()

    def browse_file(self, widget: QLineEdit, file_filter: str = "All files (*)") -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file", widget.text() or self.config.project_dir, file_filter)
        if path:
            widget.setText(path)

    def browse_folder(self, widget: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục", widget.text() or self.config.project_dir)
        if path:
            widget.setText(path)

    def select_cookie_file(self) -> None:
        self.browse_file(self.cookies_path, "JSON files (*.json)")
        self.save_fields()

    def open_project_folder(self) -> None:
        path = Path(self.project_dir.text() or self.config.project_dir)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def set_status(self, value: str, hint: str = "") -> None:
        self.system_status.setText(value)
        self.sidebar_status.setText(value)
        self.status_card.set(value, hint)

    def log(self, message: str) -> None:
        stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.log_entries.append(stamped)
        self.render_logs()

    def render_logs(self) -> None:
        if not hasattr(self, "log_box"):
            return
        selected = self.log_filter.currentText() if hasattr(self, "log_filter") else "Tất cả"
        if selected == "Tất cả":
            rows = self.log_entries
        else:
            needle = selected.upper()
            rows = [row for row in self.log_entries if needle in row.upper()]
        self.log_box.setPlainText("\n".join(rows))
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def update_metrics(self, collected: int, synced: int) -> None:
        self.local_card.set(f"{collected} bài")
        self.total_synced += synced
        self.synced_card.set(str(self.total_synced), "Tính theo phiên chạy hiện tại")
        self.sync_summary.setText(f"Phiên hiện tại đã gửi {collected} bài, ghi nhận {synced} bản ghi mới hoặc cập nhật.")

    def refresh_cookie_status(self) -> None:
        if hasattr(self, "cookies_path"):
            self.config.cookies_path = self.cookies_path.text().strip() or self.config.cookies_path
        self.cookie_status = check_cookie_file(self.config.cookies_path)
        title = self.cookie_status.title
        self.cookie_card.set(title, self.cookie_status.message)
        self.cookie_title.setText(title)
        self.cookie_message.setText(self.cookie_status.message)
        self.sidebar_cookie.setText(f"Cookie: {title}")
        if self.cookie_status.ok:
            self.overview_notice.setText("Cookie đã đọc được. Bước tiếp theo nên kiểm tra API rồi chạy cào dữ liệu.")
        else:
            self.overview_notice.setText("Cookie chưa sẵn sàng. Hãy bấm 'Lấy cookie Facebook' hoặc chọn đúng file cookies.json trước khi chạy.")
        self.log(f"INFO Cookie: {title} - {self.cookie_status.message}")

    def refresh_local_state(self) -> None:
        counts = get_local_counts(self.config)
        posts = int(counts.get("posts", 0))
        comments = int(counts.get("comments", 0))
        self.local_card.set(f"{posts} bài", f"{comments} bình luận trong SQLite")
        if counts.get("status") != "ok":
            self.log(f"WARNING Không đọc được SQLite local: {counts.get('status')}")

    def run_background(self, task) -> None:
        if self.active_thread and self.active_thread.isRunning():
            QMessageBox.warning(self, "Đang chạy", "Một tác vụ khác đang chạy. Vui lòng đợi hoàn tất hoặc dừng scraper.")
            return
        thread = QThread(self)
        worker = Worker(task)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self.log)
        worker.metrics.connect(self.update_metrics)
        worker.done.connect(self.on_task_done)
        worker.done.connect(thread.quit)
        worker.done.connect(worker.deleteLater)
        thread.finished.connect(self._clear_background_state)
        thread.finished.connect(thread.deleteLater)
        self.active_thread = thread
        self.active_worker = worker
        thread.start()

    def _clear_background_state(self) -> None:
        self.active_thread = None
        self.active_worker = None

    def on_task_done(self, ok: bool, message: str) -> None:
        if ok:
            self.set_status("Success", "Tác vụ vừa hoàn tất")
            self.log(f"INFO {message}")
        else:
            self.last_error = message
            self.set_status("Failed", "Có lỗi cần kiểm tra trong Logs")
            self.error_card.set("Có lỗi", message)
            self.log(f"ERROR {message}")
        self.refresh_local_state()
        self.refresh_cookie_status()

    def setup_env(self) -> None:
        self.save_fields()
        self.set_status("Setting up", "Đang tạo hoặc cập nhật .venv scraper")
        self.run_background(lambda emit, metrics: (setup_environment(self.config, emit), "Môi trường scraper đã sẵn sàng")[1])

    def capture_cookies(self) -> None:
        self.save_fields()
        self.set_status("Capturing cookie", "Sắp mở trình duyệt Facebook để lấy cookie")

        def task(emit, metrics):
            emit("INFO Trình duyệt Facebook sẽ mở ra. Hãy đăng nhập nếu được yêu cầu.")
            saved_path = capture_facebook_cookies(self.config, emit, timeout_seconds=600)
            return f"Đã lấy cookie Facebook thành công: {saved_path}"

        self.run_background(task)

    def run_scraper(self) -> None:
        self.save_fields()
        if not self.cookie_status.ok:
            QMessageBox.warning(self, "Thiếu cookie", f"{self.cookie_status.title}\n\n{self.cookie_status.message}")
            self.set_status("Cookie missing", "Cần cookies.json hợp lệ trước khi cào")
            return
        self.set_status("Running", "Đang cào dữ liệu từ Facebook")
        self.last_run_card.set(datetime.now().strftime("%Y-%m-%d %H:%M"), "Scraper vừa được gọi")

        def task(emit, metrics):
            self.scraper = ScraperProcess(self.config, emit)
            code = self.scraper.run()
            if code != 0:
                raise RuntimeError(f"Scraper exited with code {code}")
            emit("INFO Scraper completed.")
            if self.config.auto_sync_after_scrape:
                emit("INFO Auto-sync đang bật. Chuẩn bị đồng bộ dữ liệu.")
                result = self._sync_payload(emit, metrics)
                return f"Scraper hoàn tất và đã đồng bộ: {result}"
            return "Scraper hoàn tất"

        self.run_background(task)

    def resume_scraper(self) -> None:
        self.log("INFO Tiếp tục cào dữ liệu từ bài mới nhất. Scraper sẽ tự dừng khi gặp đủ ngưỡng bài trùng đã cấu hình.")
        self.run_scraper()

    def stop_scraper(self) -> None:
        if self.scraper:
            self.scraper.stop()
            self.set_status("Stopped", "Đã gửi tín hiệu dừng scraper")
            self.log("WARNING Đã yêu cầu dừng scraper.")

    def _sync_payload(self, emit, metrics) -> dict:
        token = get_api_token()
        if not token:
            raise RuntimeError("API token đang trống. Hãy nhập token ở trang Cấu hình.")
        payload = build_ingest_batch(self.config)
        emit(f"INFO Prepared batch {payload['batch_id']} with {len(payload['posts'])} posts")
        client = SyncClient(self.config, token)
        result = client.send_batch(payload)
        synced = int(result.get("inserted", 0)) + int(result.get("updated", 0))
        metrics(len(payload["posts"]), synced)
        return result

    def sync_now(self) -> None:
        self.save_fields()
        self.set_status("Syncing", "Đang gửi dữ liệu lên backend")

        def task(emit, metrics):
            result = self._sync_payload(emit, metrics)
            return f"Sync result: {result}"

        self.run_background(task)

    def test_api(self) -> None:
        self.save_fields()
        self.set_status("Testing API", "Đang kiểm tra backend")

        def task(emit, metrics):
            token = get_api_token()
            if not token:
                raise RuntimeError("API token đang trống. Hãy nhập token ở trang Cấu hình.")
            emit(f"INFO Testing backend: {self.config.backend_url}")
            result = SyncClient(self.config, token).test_connection()
            emit(f"INFO API health: {result}")
            return "API connection OK"

        self.run_background(task)

    def _apply_schedule(self):
        self.scheduler.remove_all_jobs()
        raw = self.schedule_value.text().strip()
        if raw.startswith("hours:"):
            hours = int(raw.split(":", 1)[1])
            return self.scheduler.add_job(self.schedule_run_requested.emit, "interval", hours=hours, id="ctsv_scraper_interval")
        if raw.startswith("daily:"):
            time_part = raw.split(":", 1)[1]
            hour, minute = [int(part) for part in time_part.split(":")]
            return self.scheduler.add_job(self.schedule_run_requested.emit, "cron", hour=hour, minute=minute, id="ctsv_scraper")
        raise ValueError("Dùng daily:22:00 hoặc hours:4")

    def enable_schedule(self) -> None:
        self.save_fields()
        try:
            job = self._apply_schedule()
        except Exception as exc:
            QMessageBox.warning(self, "Sai định dạng", str(exc))
            return
        self.config.schedule_enabled = True
        save_config(self.config)
        next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M") if job.next_run_time else "chưa xác định"
        self.schedule_status.setText(f"Đã bật lịch: {raw}. Lần chạy kế tiếp: {next_run}")
        self.set_status("Scheduled", "Scheduler đang bật")
        self.log(f"INFO Schedule enabled: {raw}, next run: {next_run}")

    def restore_schedule_if_enabled(self) -> None:
        if not self.config.schedule_enabled:
            return
        try:
            job = self._apply_schedule()
        except Exception as exc:
            self.schedule_status.setText(f"Lịch đã lưu không hợp lệ: {exc}")
            self.log(f"WARNING Stored schedule is invalid: {exc}")
            return
        next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M") if job.next_run_time else "chưa xác định"
        self.schedule_status.setText(f"Đang chạy lịch: {self.schedule_value.text().strip()}. Lần chạy kế tiếp: {next_run}")
        self.set_status("Scheduled", "Scheduler đang bật")
        self.log(f"INFO Schedule restored, next run: {next_run}")

    def disable_schedule(self) -> None:
        self.scheduler.remove_all_jobs()
        self.config.schedule_enabled = False
        save_config(self.config)
        self.schedule_status.setText("Đã tắt lịch chạy")
        self.set_status("Idle", "Scheduler đã tắt")
        self.log("INFO Schedule disabled.")

    def copy_logs(self) -> None:
        QApplication.clipboard().setText("\n".join(self.log_entries))
        self.log("INFO Đã copy log vào clipboard.")

    def export_logs(self) -> None:
        default = f"ctsv-scraper-log-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Export log", default, "Text files (*.txt)")
        if path:
            Path(path).write_text("\n".join(self.log_entries), encoding="utf-8")
            self.log(f"INFO Đã export log: {path}")

    def _build_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.log("WARNING System tray không khả dụng trên máy này.")
            return
        icon = build_app_icon(64)
        self.tray_icon = QSystemTrayIcon(icon, self)
        self.tray_icon.setToolTip("CTSV Scraper Control Center")

        menu = QMenu(self)
        show_action = QAction("Mở CTSV Scraper", self)
        show_action.triggered.connect(self.show_from_tray)
        run_action = QAction("Chạy cào dữ liệu", self)
        run_action.triggered.connect(self.run_scraper)
        resume_action = QAction("Tiếp tục cào", self)
        resume_action.triggered.connect(self.resume_scraper)
        quit_action = QAction("Thoát hẳn", self)
        quit_action.triggered.connect(self.quit_from_tray)
        menu.addAction(show_action)
        menu.addAction(run_action)
        menu.addAction(resume_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def on_tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_from_tray()

    def show_from_tray(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def quit_from_tray(self) -> None:
        self.allow_quit = True
        QApplication.quit()

    def closeEvent(self, event) -> None:
        if not self.allow_quit and self.tray_icon and self.tray_icon.isVisible():
            self.hide()
            self.tray_icon.showMessage(
                "CTSV Scraper vẫn đang chạy",
                "Ứng dụng đã thu nhỏ xuống system tray để tiếp tục chạy lịch cào tự động.",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )
            event.ignore()
            return
        try:
            if self.active_thread and self.active_thread.isRunning():
                self.log("WARNING Ứng dụng đang đóng khi còn tác vụ nền. Đang chờ tác vụ kết thúc.")
                self.active_thread.quit()
                self.active_thread.wait(3000)
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(
        """
        QWidget#Root { background: #f4f7f3; color: #1f2937; font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 14px; }
        QWidget#Sidebar { background: #f8fbf8; border-right: 1px solid #dbe6dc; min-width: 258px; max-width: 258px; }
        QFrame#BrandPanel { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #eef7f1); border: 1px solid #d7e5da; border-radius: 16px; }
        QLabel#Brand { color: #0d5c57; font-size: 30px; font-weight: 820; line-height: 1.05; }
        QLabel#BrandDesc { color: #5f6f68; font-size: 12px; }
        QLabel#SidebarPill { background: #dff2e7; color: #0f5e4f; border: 1px solid #c7e6d3; border-radius: 12px; padding: 10px 12px; font-weight: 760; }
        QLabel#SidebarSubPill { background: #fff5df; color: #8a5a12; border: 1px solid #f2e2b9; border-radius: 12px; padding: 10px 12px; }
        QLabel#SidebarFooter { color: #74827b; font-size: 12px; }
        QListWidget#Navigation { border: 0; background: transparent; outline: none; padding-top: 4px; }
        QListWidget#Navigation::item { padding: 13px 14px; border-radius: 12px; color: #31443d; font-weight: 680; }
        QListWidget#Navigation::item:selected { background: #ffffff; color: #0d5c57; border: 1px solid #d8e6db; }
        QListWidget#Navigation::item:hover { background: #edf4ef; }
        QFrame#Header { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffffff, stop:1 #f2f7f3); border: 1px solid #dbe6dc; border-radius: 18px; }
        QLabel#HeaderTitle { font-size: 30px; font-weight: 840; color: #10211d; }
        QLabel#HeaderSubtitle, QLabel#PageSubtitle, QLabel#PanelSubtitle, QLabel#CardHint { color: #61726c; }
        QLabel#PageTitle { font-size: 26px; font-weight: 830; color: #11221f; }
        QLabel#PanelTitle { font-size: 18px; font-weight: 800; color: #11221f; }
        QLabel#CardTitle { color: #536660; font-weight: 720; }
        QLabel#CardValue { color: #10211d; font-size: 24px; font-weight: 840; }
        QLabel#CookieTitle { color: #0d5c57; font-size: 21px; font-weight: 820; }
        QLabel#GuideLine { color: #33443f; padding: 2px 0; }
        QLabel#GuideChip { background: #eef4ef; color: #31443d; border: 1px solid #dae5dc; border-radius: 10px; padding: 8px 12px; font-weight: 700; }
        QLabel#FieldLabel { color: #33443f; font-weight: 700; }
        QLabel#StatusChip { background: #10211d; color: #ffffff; border-radius: 14px; padding: 10px 16px; font-weight: 820; min-width: 80px; }
        QFrame#StatusCard, QFrame#Panel { background: #ffffff; border: 1px solid #dbe6dc; border-radius: 16px; }
        QPushButton { border: 0; border-radius: 12px; padding: 11px 16px; font-weight: 760; min-height: 22px; }
        QPushButton#PrimaryButton { background: #177e74; color: #ffffff; }
        QPushButton#PrimaryButton:hover { background: #11685f; }
        QPushButton#GhostButton { background: #edf2f7; color: #1f2937; }
        QPushButton#GhostButton:hover { background: #e1e9f0; }
        QPushButton#DangerButton { background: #fde8e8; color: #9a2424; }
        QPushButton#DangerButton:hover { background: #fbd5d5; }
        QPushButton#SmallButton { background: #eef3f1; color: #33443f; padding: 9px 14px; }
        QLineEdit, QTextEdit, QSpinBox, QComboBox { background: #ffffff; border: 1px solid #cedbd1; border-radius: 12px; padding: 9px 11px; color: #17212d; }
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus { border: 1px solid #177e74; }
        QStackedWidget#PageStack, QScrollArea#PageScroll, QWidget#PageViewport, QWidget#PageBody { background: transparent; border: 0; }
        QCheckBox { color: #33443f; font-weight: 650; spacing: 8px; }
        """
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
