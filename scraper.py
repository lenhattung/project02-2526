from playwright.sync_api import sync_playwright
import json
import sqlite3
import os
import re
from datetime import datetime

LINKS_FILE = "link.txt"           # danh sách URL, mỗi dòng 1 trang (bỏ qua dòng trống / #)
DEFAULT_URL = "https://www.facebook.com/DNTUConfession/"
MAX_POSTS_PER_PAGE = 2500         # tối đa số bài cào mỗi trang
SCRAPE_COMMENTS = 1           # 1 = cào comment của mỗi bài, 0 = không cào
MAX_COMMENT_EXPANDS = 50      # >0 = mở popup bình luận 1 lần rồi lấy hết comment đang hiển thị (không cuộn thêm); 0 = không cào
DB_FILE = "data/posts.db"

# --- Tốc độ cào (ms) — tăng lên nếu mạng chậm / bài chưa kịp load ra ---
SCROLL_STEP_RATIO = 0.5    # mỗi lần cuộn = bao nhiêu phần màn hình (nhỏ hơn = cuộn chậm, kỹ hơn)
SCROLL_WAIT_MS = 5000      # chờ sau mỗi lần cuộn để FB render bài mới
POST_RENDER_WAIT_MS = 2000 # chờ sau khi cuộn tới 1 bài để bài render đầy đủ
EXPAND_WAIT_MS = 2000      # chờ sau khi bấm 'Xem thêm' của bài
BOTTOM_KICK_WAIT_MS = 3500 # chờ ở đáy sau khi 'giật' để FB load thêm


def load_links():
    """Đọc danh sách URL từ LINKS_FILE; fallback về DEFAULT_URL nếu không có."""
    if not os.path.exists(LINKS_FILE):
        print(f"⚠️ Không thấy {LINKS_FILE} → dùng URL mặc định")
        return [DEFAULT_URL]
    urls = []
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                urls.append(s)
    return urls or [DEFAULT_URL]

def _ensure_columns(cursor, table, columns):
    """Thêm cột còn thiếu vào bảng đã tồn tại (migration cho DB cũ)."""
    existing = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
            print(f"🔧 Migration: thêm cột '{name}' vào bảng '{table}'")


def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            source TEXT,
            posted_at TEXT,
            like_count TEXT,
            comment_count TEXT,
            collected_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            content TEXT,
            source TEXT,
            collected_at TEXT,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )
    """)
    # Migration: bổ sung cột cho DB đã tạo từ trước
    _ensure_columns(cursor, "posts", {
        "source": "source TEXT",
        "posted_at": "posted_at TEXT",
        "like_count": "like_count TEXT",
        "comment_count": "comment_count TEXT",
    })
    _ensure_columns(cursor, "comments", {"source": "source TEXT"})
    conn.commit()
    return conn

def load_cookies(context):
    with open("cookies.json", "r") as f:
        cookies = json.load(f)

    # Fix sameSite value không hợp lệ
    for cookie in cookies:
        same_site = cookie.get("sameSite", "None")
        if same_site not in ["Strict", "Lax", "None"]:
            cookie["sameSite"] = "None"

    context.add_cookies(cookies)
    print("✅ Đã load cookies")

def check_logged_in(page):
    try:
        page.goto("https://www.facebook.com", timeout=60000)
        page.wait_for_load_state("domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        login_btn = page.query_selector("input[name='email']")
        if login_btn:
            print("❌ Cookies hết hạn! Cần chạy lại login.py")
            return False

        print("✅ Đã đăng nhập thành công")
        return True
    except Exception as e:
        print(f"⚠️ Lỗi kiểm tra: {e}")
        return False


def close_login_popup(page):
    try:
        close_btn = page.query_selector("div[aria-label='Close']")
        if close_btn:
            close_btn.click()
            page.wait_for_timeout(1000)
            print("✅ Đã đóng popup")
    except:
        pass

# Tên page + các dòng "rác" cần loại khỏi nội dung bài
PAGE_NAME = "DNTU - Confession"
NOISE_LINES = {
    PAGE_NAME,
    "Thu gọn", "See less",
    "Xem thêm", "See more",
    "Xem thêm bình luận", "View more comments",
    "Tất cả cảm xúc:", "All reactions:",
    "Thích", "Like", "Bình luận", "Comment", "Chia sẻ", "Share",
}

# Selector vỏ bọc mỗi bài viết trong feed (xác định qua DevTools: div[aria-posinset]).
POST_SELECTOR = "div[aria-posinset]"

# JS chạy trong trình duyệt: trích nội dung chính của 1 bài (div[aria-posinset]).
#  - Gom các khối div[dir="auto"] NGOÀI CÙNG.
#  - Loại các article con (comment lồng nhau) và khối dir="auto" lồng nhau.
#  - Đọc cả emoji (img alt) và xuống dòng (br).
_EXTRACT_JS = r'''
(el) => {
  const nested = Array.from(el.querySelectorAll('div[role="article"]'));
  const inNested = (node) => {
    let p = node.parentElement;
    while (p && p !== el) { if (nested.indexOf(p) !== -1) return true; p = p.parentElement; }
    return false;
  };
  const inAutoDiv = (node) => {
    let p = node.parentElement;
    while (p && p !== el) {
      if (p.tagName === 'DIV' && p.getAttribute('dir') === 'auto') return true;
      p = p.parentElement;
    }
    return false;
  };
  const getText = (node) => {
    let out = '';
    node.childNodes.forEach((c) => {
      if (c.nodeType === 3) out += c.textContent;          // text node
      else if (c.nodeName === 'IMG') out += (c.getAttribute('alt') || '');  // emoji
      else if (c.nodeName === 'BR') out += '\n';
      else out += getText(c);
    });
    return out;
  };

  const parts = [];
  for (const b of Array.from(el.querySelectorAll('div[dir="auto"]'))) {
    if (inNested(b) || inAutoDiv(b)) continue;   // bỏ comment & khối lồng nhau
    const t = getText(b).trim();
    if (t) parts.push(t);
  }
  return parts.join('\n').trim();
}
'''

# Dòng ngày/tháng ở header bài (vd "4 tháng 11, 2025") cần loại bỏ
_DATE_RE = re.compile(r'^\d{1,2}\s*(tháng|thg)\b', re.IGNORECASE)
# Ô soạn bình luận (vd "Bình luận dưới tên Le Nhat Tung")
_COMPOSER_PREFIXES = ("Bình luận dưới tên", "Viết bình luận", "Comment as", "Write a comment")


def _clean_content(text):
    """Loại các dòng rác (tên page, ngày tháng, nút bấm, ô bình luận...) khỏi nội dung."""
    lines = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s or s in NOISE_LINES:
            continue
        if s.isdigit():                      # số đếm reaction/comment
            continue
        if _DATE_RE.match(s):                # dòng ngày tháng
            continue
        if s.startswith(_COMPOSER_PREFIXES): # ô soạn bình luận
            continue
        lines.append(s)
    return "\n".join(lines).strip()


def close_any_dialog(page):
    """Đóng mọi popup/dialog đang mở (Escape, fallback nút Đóng)."""
    try:
        if not page.query_selector("div[role='dialog']"):
            return
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
        if page.query_selector("div[role='dialog']"):
            btn = (
                page.query_selector("div[role='dialog'] div[aria-label='Đóng']") or
                page.query_selector("div[role='dialog'] div[aria-label='Close']")
            )
            if btn:
                btn.click()
                page.wait_for_timeout(400)
    except Exception:
        pass


def expand_see_more_inline(post, page):
    """Click 'Xem thêm' của BÀI để mở rộng nội dung tại chỗ (inline).

    Dùng :text-is (khớp CHÍNH XÁC) để tránh khớp nhầm 'Xem thêm bình luận'
    — vốn mở popup bình luận và gây nội dung lặp.
    """
    try:
        see_more = (
            post.query_selector("div[role='button']:text-is('Xem thêm')") or
            post.query_selector("div[role='button']:text-is('See more')")
        )
        if see_more:
            see_more.click()
            page.wait_for_timeout(EXPAND_WAIT_MS)
            return True
    except Exception:
        pass
    return False

def extract_post_content(post):
    """Lấy nội dung chính của 1 status đang hiển thị.

    Dùng JS gom các khối div[dir='auto'] ngoài cùng, bỏ qua comment.
    Trả về "" nếu là comment hoặc không có nội dung → bỏ qua.
    """
    try:
        raw = post.evaluate(_EXTRACT_JS) or ""
    except Exception:
        return ""
    return _clean_content(raw)


# JS lấy ngày đăng: tìm link timestamp trong header (text dạng "4 tháng 11, 2025",
# "12 giờ", "2 ngày", "Vừa xong"...). Lấy link ĐẦU TIÊN khớp mẫu ngày/giờ.
_EXTRACT_DATE_JS = r'''
(el) => {
  const re = /tháng|giờ|phút|ngày|tuần|năm|vừa xong|hôm qua|yesterday|just now|\b\d{4}\b|\b\d+\s*[hmdwy]\b/i;
  const links = Array.from(el.querySelectorAll('a[role="link"], a[href]'));
  for (const a of links) {
    const t = (a.innerText || '').trim();
    if (t && t.length <= 40 && re.test(t)) return t;
  }
  return '';
}
'''


def extract_post_date(post):
    """Lấy text ngày đăng từ link timestamp trong header bài."""
    try:
        return (post.evaluate(_EXTRACT_DATE_JS) or "").strip()
    except Exception:
        return ""


# JS lấy SỐ LƯỢNG cảm xúc (like) và số bình luận của 1 bài.
#  - Số bình luận: text dạng "12 bình luận" / "3 comments" (rất ổn định).
#  - Số cảm xúc: số nằm cạnh cụm icon cảm xúc (img alt = tên cảm xúc).
#  - Giữ nguyên dạng rút gọn của FB (vd "1,2K"). Trả "" nếu không thấy.
_EXTRACT_STATS_JS = r'''
(el) => {
  const norm = (s) => (s || '').replace(/ /g, ' ').replace(/\s+/g, ' ').trim();
  const ONLY_NUM = /^[\d][\d.,]*\s?[KkMmNnTr]*$/;     // 12 / 1,2K / 3.4N ...
  const REACTS = /(Thích|Yêu thích|Thương thương|Haha|Wow|Buồn|Phẫn nộ|Like|Love|Care|Sad|Angry)/i;

  // --- Số bình luận ---
  let comment = '';
  for (const n of Array.from(el.querySelectorAll('span, div[role="button"], a[role="link"]'))) {
    const m = norm(n.innerText).match(/([\d][\d.,]*\s?[KkMmNnTr]*)\s*(bình luận|comments?)\b/i);
    if (m) { comment = m[1].trim(); break; }
  }

  // --- Số cảm xúc (like): tìm số gần cụm icon cảm xúc ---
  let like = '';
  const imgs = Array.from(el.querySelectorAll('img[alt]'))
      .filter(im => REACTS.test(im.getAttribute('alt') || ''));
  for (const im of imgs) {
    let p = im;
    for (let i = 0; i < 6 && p; i++) {
      p = p.parentElement;
      if (!p) break;
      const cand = Array.from(p.querySelectorAll('span'))
          .map(s => norm(s.innerText))
          .find(t => ONLY_NUM.test(t));
      if (cand) { like = cand; break; }
    }
    if (like) break;
  }
  // Fallback: aria-label kiểu "Tất cả cảm xúc: X"
  if (!like) {
    for (const a of Array.from(el.querySelectorAll('[aria-label]'))) {
      const m = norm(a.getAttribute('aria-label'))
          .match(/(?:Tất cả cảm xúc|All reactions)[:\s]*([\d][\d.,]*\s?[KkMmNnTr]*)/i);
      if (m) { like = m[1].trim(); break; }
    }
  }

  return { like, comment };
}
'''


def extract_post_stats(post):
    """Lấy (số like, số comment) của bài dưới dạng text. Trả ('', '') nếu lỗi."""
    try:
        stats = post.evaluate(_EXTRACT_STATS_JS) or {}
        return (stats.get("like") or "").strip(), (stats.get("comment") or "").strip()
    except Exception:
        return "", ""


def scrape_page(page, conn, page_url, max_posts=MAX_POSTS_PER_PAGE):
    print(f"\n🌐 Đang vào trang {page_url}")
    page.goto(page_url, timeout=60000)
    page.wait_for_load_state("domcontentloaded", timeout=60000)
    page.wait_for_timeout(4000)

    close_login_popup(page)
    page.wait_for_timeout(2000)

    cursor = conn.cursor()

    collected_at = datetime.now().isoformat()
    saved_count = 0
    seen_contents = set()  # prefix nội dung đã xử lý trong phiên này (lọc nhanh)

    # Nạp toàn bộ nội dung đã có trong DB (các lần cào trước) để bỏ qua bài trùng 100%
    existing_contents = set(row[0] for row in cursor.execute("SELECT content FROM posts"))
    print(f"📚 DB hiện có {len(existing_contents)} bài — sẽ bỏ qua bài trùng hoàn toàn")

    # Mở file txt 1 lần, ghi từng bài ngay khi có (append để giữ lịch sử các phiên)
    os.makedirs("data", exist_ok=True)
    txt_file = open("data/data.txt", "a", encoding="utf-8")
    txt_file.write(f"\n===== Phiên cào: {collected_at} | {page_url} =====\n")

    print("📜 Đang scroll và thu thập bài viết...")
    last_height = page.evaluate("document.body ? document.body.scrollHeight : 0")
    scroll_count = 0
    no_new_count = 0       # Đếm số lần không có bài mới
    reached_limit = False  # Đã đạt giới hạn max_posts chưa

    # So khớp "đang ở đúng trang cần cào" (bỏ dấu / cuối). Nếu page.url không bắt
    # đầu bằng chuỗi này nghĩa là đã trôi sang trang khác (vd facebook.com home).
    page_url_base = page_url.rstrip("/")

    # Cào đến khi hết nội dung (no_new_count >= 3) hoặc đạt max_posts
    while True:

        # AN TOÀN: nếu trang đã rời khỏi page_url (vd bị nhảy về facebook.com home
        # khi rơi vào đáy / sau khi cào comment) → quay lại đúng trang rồi cuộn tiếp.
        # Dedup (existing_contents/seen_contents) đảm bảo không lưu trùng khi cuộn lại.
        if not page.url.rstrip("/").startswith(page_url_base):
            print(f"\n⚠️ Trang đã rời {page_url} (hiện: {page.url}) → quay lại trang cần cào")
            try:
                page.goto(page_url, timeout=60000)
                page.wait_for_load_state("domcontentloaded", timeout=60000)
                page.wait_for_timeout(3000)
                close_login_popup(page)
            except Exception as e:
                print(f"⚠️ Lỗi khi quay lại {page_url}: {e}")
                page.wait_for_timeout(2000)
            last_height = page.evaluate("document.body ? document.body.scrollHeight : 0")
            no_new_count = 0

        # Lấy tất cả bài hiện có trên trang (mỗi bài = 1 div[aria-posinset])
        posts = page.query_selector_all(POST_SELECTOR)

        for i, post in enumerate(posts):
            try:
                # Trích nhanh nội dung (chưa mở 'Xem thêm') để lọc & chống trùng.
                # article nào không có story_message (comment, quảng cáo) → bỏ qua.
                preview = extract_post_content(post)
                if not preview or len(preview) < 10:
                    continue

                # Dedup theo NỘI DUNG bài (ổn định hơn innerHTML).
                # Dùng prefix vì bản rút gọn và bản đầy đủ có chung phần đầu.
                content_key = preview[:80]
                if content_key in seen_contents:
                    continue

                # Scroll đến bài này để Facebook render đầy đủ nội dung
                post.scroll_into_view_if_needed()
                page.wait_for_timeout(POST_RENDER_WAIT_MS)

                # Mở rộng 'Xem thêm' ngay tại chỗ (nếu có) rồi lấy nội dung đầy đủ
                if expand_see_more_inline(post, page):
                    print("   🔽 Đã mở rộng 'Xem thêm'")

                # Phòng trường hợp lỡ mở popup → đóng lại trước khi tiếp tục
                close_any_dialog(page)

                content = extract_post_content(post) or preview

                if not content or len(content) < 10:
                    # Bài CHƯA render đầy đủ (hoặc phần tử vừa bị FB tái dựng) →
                    # KHÔNG đánh dấu seen để còn lấy lại ở vòng cuộn sau.
                    continue

                # Đến đây mới chắc chắn có nội dung hợp lệ → giờ mới đánh dấu đã xử lý
                # (đánh dấu cả prefix preview lẫn prefix nội dung đầy đủ để dedup chắc chắn).
                seen_contents.add(content_key)
                seen_contents.add(content[:80])

                # Bài đã có trong DB (cào lần trước hoặc phiên này) hay là bài mới?
                is_old = content in existing_contents

                # LUÔN in toàn bộ nội dung ra để ĐỐI CHIẾU với bài đang hiển thị trên màn hình
                # (cả bài MỚI lẫn bài CŨ), bất kể có lưu xuống DB hay không.
                tag = "CŨ (đã có)" if is_old else "MỚI"
                print("\n" + "─" * 70)
                print(f"📄 [{tag}] nội dung bài đang hiển thị (đối chiếu màn hình):")
                print(content)
                print("─" * 70)

                # Bỏ qua nếu nội dung TRÙNG 100% với bài đã có
                if is_old:
                    continue
                existing_contents.add(content)

                # Lấy ngày đăng + số like/comment từ bài
                posted_at = extract_post_date(post)
                like_count, comment_count = extract_post_stats(post)

                # Lưu NGAY vào DB + commit để không mất nếu lỗi giữa chừng
                cursor.execute(
                    "INSERT INTO posts (content, source, posted_at, like_count, comment_count, collected_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (content, page_url, posted_at, like_count, comment_count, collected_at)
                )
                conn.commit()
                post_id = cursor.lastrowid
                saved_count += 1
                print(f"✅ [{post_id}] ({posted_at or '?'}) 👍{like_count or '0'} 💬{comment_count or '0'} | {content[:45]}...")

                # Ghi NGAY vào file txt
                txt_file.write(
                    f"--- Bài {post_id} | đăng: {posted_at} | 👍 {like_count} | 💬 {comment_count} | cào: {collected_at} ---\n{content}\n\n"
                )
                txt_file.flush()

                # Cào comment của bài này (nếu bật) — lưu kèm post_id
                if SCRAPE_COMMENTS:
                    n_cmt = scrape_comments(post, page, conn, post_id, page_url, collected_at)
                    if n_cmt:
                        print(f"   💬 {n_cmt} bình luận")

                # Đạt giới hạn số bài cho trang này → dừng
                if saved_count >= max_posts:
                    print(f"🛑 Đã đạt giới hạn {max_posts} bài cho trang này")
                    reached_limit = True
                    break

            except Exception as e:
                print(f"⚠️ Lỗi bài {i+1}: {e}")

        if reached_limit:
            break

        # Commit định kỳ để không mất dữ liệu nếu lỗi giữa chừng
        conn.commit()

        # DI CHUYỂN trang xuống MỘT ĐOẠN (scrollBy) — luôn tiến tới, không đứng yên một chỗ
        # (guard document.body vì trang có thể đang điều hướng/load lại → body null)
        view_h = page.evaluate("window.innerHeight") or 800
        try:
            page.evaluate(f"window.scrollBy(0, {int(view_h * SCROLL_STEP_RATIO)})")
        except Exception as e:
            print(f"\n⚠️ Lỗi scroll (trang đang load lại?): {e}")
            page.wait_for_timeout(2000)
            scroll_count += 1
            continue
        page.wait_for_timeout(SCROLL_WAIT_MS)

        # Đo lại chiều cao trang & kiểm tra đã tới đáy chưa
        new_height = page.evaluate("document.body ? document.body.scrollHeight : 0")
        at_bottom = page.evaluate(
            "(window.innerHeight + window.scrollY) >= ((document.body ? document.body.scrollHeight : 0) - 150)"
        )

        if new_height > last_height:
            # Có nội dung mới được load → tiếp tục
            print(f"\n🔄 Scroll {scroll_count+1} | Height: {last_height} → {new_height}px")
            last_height = new_height
            no_new_count = 0
        elif at_bottom:
            # Đã ở ĐÁY mà chưa có nội dung mới → GIẬT (lên rồi xuống) để kích Facebook load thêm
            no_new_count += 1
            print(f"\n⏳ Ở đáy, chưa có nội dung mới ({no_new_count}/5) — đang kích load...")
            page.evaluate("window.scrollBy(0, -400)")
            page.wait_for_timeout(800)
            page.evaluate("window.scrollTo(0, document.body ? document.body.scrollHeight : 0)")
            page.wait_for_timeout(BOTTOM_KICK_WAIT_MS)
            new_height = page.evaluate("document.body ? document.body.scrollHeight : 0")
            if new_height > last_height:
                last_height = new_height
                no_new_count = 0
            elif no_new_count >= 5000:
                print("✅ Đã đến cuối trang!")
                break
        else:
            # Chưa tới đáy → vẫn đang di chuyển qua phần đã load, đi tiếp
            no_new_count = 0

        scroll_count += 1

    conn.commit()

    txt_file.write(f"# Phiên {collected_at}: {saved_count} bài mới\n")
    txt_file.close()

    print(f"\n✅ Hoàn tất! Đã lưu {saved_count} bài viết MỚI vào:")
    print(f"   📁 {DB_FILE}")
    print(f"   📄 data/data.txt")

# JS trích các comment trong 1 bài (div[aria-posinset]).
#  - Comment = div[role='article'] có aria-label "Bình luận"/"Phản hồi"/"Comment"/"Reply".
#  - Với mỗi comment, lấy text các div[dir="auto"] ngoài cùng, loại text của reply lồng nhau.
#  - Đọc cả emoji (img alt).
_EXTRACT_COMMENTS_JS = r'''
(el) => {
  const getText = (node) => {
    let out = '';
    node.childNodes.forEach((c) => {
      if (c.nodeType === 3) out += c.textContent;
      else if (c.nodeName === 'IMG') out += (c.getAttribute('alt') || '');
      else if (c.nodeName === 'BR') out += '\n';
      else out += getText(c);
    });
    return out;
  };
  const isComment = (a) => /Bình luận|Phản hồi|Comment|Reply/i.test(a.getAttribute('aria-label') || '');

  const results = [];
  for (const a of Array.from(el.querySelectorAll('div[role="article"]'))) {
    if (!isComment(a)) continue;
    const deeper = Array.from(a.querySelectorAll('div[role="article"]'));  // reply lồng nhau
    const inDeeper = (node) => {
      let p = node.parentElement;
      while (p && p !== a) { if (deeper.indexOf(p) !== -1) return true; p = p.parentElement; }
      return false;
    };
    const inAutoDiv = (node) => {
      let p = node.parentElement;
      while (p && p !== a) {
        if (p.tagName === 'DIV' && p.getAttribute('dir') === 'auto') return true;
        p = p.parentElement;
      }
      return false;
    };
    const parts = [];
    for (const b of Array.from(a.querySelectorAll('div[dir="auto"]'))) {
      if (inDeeper(b) || inAutoDiv(b)) continue;
      const t = getText(b).trim();
      if (t) parts.push(t);
    }
    const text = parts.join('\n').trim();
    if (text) results.push(text);
  }
  return results;
}
'''


def open_comment_popup(post, page):
    """Mở phần bình luận của bài (click 'Bình luận' / 'Xem thêm bình luận').

    Trả về vùng chứa comment để cào:
      - div[role='dialog'] nếu mở ra popup, hoặc
      - div[role='main'] nếu click mở sang trang permalink, hoặc
      - chính `post` (inline) nếu không có gì đổi.
    """
    js = r'''
    (el) => {
      const cands = Array.from(el.querySelectorAll('div[role="button"], span[role="button"], a[role="link"]'));
      // Ưu tiên mở rộng/xem thêm bình luận, fallback nút "Bình luận"
      const b = cands.find(x => /Xem thêm.*bình luận|bình luận khác|xem các bình luận|more comments|previous comments/i.test(x.innerText || ''))
             || cands.find(x => /^(Bình luận|Comment)$/i.test((x.innerText || '').trim()));
      if (b) { b.scrollIntoView({block: 'center'}); b.click(); return true; }
      return false;
    }
    '''
    try:
        clicked = post.evaluate(js)
    except Exception:
        return post
    if not clicked:
        return post
    page.wait_for_timeout(2500)
    dialog = page.query_selector("div[role='dialog']")
    if dialog:
        return dialog
    # Không có dialog → có thể đã mở permalink (đổi URL): cào ở vùng main
    main = page.query_selector("div[role='main']")
    return main or post


# JS: trong vùng bình luận, bấm các nút "Xem thêm bình luận / phản hồi" để load thêm.
_EXPAND_MORE_JS = r'''
(el) => {
  const cands = Array.from(el.querySelectorAll('div[role="button"], span[role="button"]'));
  let n = 0;
  for (const x of cands) {
    const t = (x.innerText || '').trim();
    if (/Xem thêm.*bình luận|bình luận trước|bình luận khác|xem các bình luận|xem thêm.*phản hồi|phản hồi$|more comments|previous comments|view.*repl/i.test(t)) {
      x.scrollIntoView({block: 'center'}); x.click(); n++;
    }
  }
  return n;
}
'''

# JS: cuộn vùng cuộn được cao nhất bên trong el xuống đáy (để FB load thêm comment).
_SCROLL_AREA_JS = r'''
(el) => {
  let best = el, bestH = el.scrollHeight || 0;
  for (const n of Array.from(el.querySelectorAll('*'))) {
    const st = getComputedStyle(n);
    if ((st.overflowY === 'auto' || st.overflowY === 'scroll') && n.scrollHeight > n.clientHeight + 20) {
      if (n.scrollHeight > bestH) { bestH = n.scrollHeight; best = n; }
    }
  }
  best.scrollTop = best.scrollHeight;
  return best.scrollHeight;
}
'''


def load_more_comments(container, page, rounds=MAX_COMMENT_EXPANDS):
    """Cuộn + bấm 'xem thêm bình luận' trong vùng comment để load thêm, tối đa `rounds` vòng.

    Tự dừng khi không còn comment mới (chiều cao vùng cuộn không tăng nữa).
    """
    last_h = 0
    for _ in range(rounds):
        try:
            container.evaluate(_EXPAND_MORE_JS)
            h = container.evaluate(_SCROLL_AREA_JS)
        except Exception:
            break
        page.wait_for_timeout(1500)
        if h and h <= last_h:   # không load thêm được nữa → dừng
            break
        last_h = h


def scrape_comments(post, page, conn, post_id, source, collected_at):
    """Cào comment của 1 bài, lưu kèm post_id. Trả về số comment đã lưu.

    Mở phần bình luận, cuộn/expand để load thêm (MAX_COMMENT_EXPANDS vòng), lấy hết
    comment đang có, rồi đóng popup / quay lại trang feed ban đầu.
    """
    url_before = page.url
    container = open_comment_popup(post, page) if MAX_COMMENT_EXPANDS > 0 else post

    # Load thêm comment (cuộn + bấm xem thêm) trong vùng vừa mở
    if MAX_COMMENT_EXPANDS > 0:
        load_more_comments(container, page)

    try:
        raw_comments = container.evaluate(_EXTRACT_COMMENTS_JS) or []
    except Exception:
        raw_comments = []

    cursor = conn.cursor()
    seen = set()
    count = 0
    for raw in raw_comments:
        text = _clean_content(raw)
        if not text or len(text) < 2 or text in seen:
            continue
        seen.add(text)
        cursor.execute(
            "INSERT INTO comments (post_id, content, source, collected_at) VALUES (?, ?, ?, ?)",
            (post_id, text, source, collected_at)
        )
        count += 1
    conn.commit()

    # Đóng popup (nếu có) rồi KHÔI PHỤC trang feed ban đầu nếu đã rời (vd mở permalink)
    close_any_dialog(page)
    if page.url != url_before:
        # Thử go_back trước; trên SPA của FB go_back có thể "thành công" nhưng
        # lại nhảy về facebook.com (home feed) → PHẢI kiểm tra lại URL.
        try:
            page.go_back(timeout=30000)
            page.wait_for_timeout(1500)
        except Exception:
            pass
        # Nếu vẫn chưa quay đúng về feed ban đầu → ép goto lại
        if page.url != url_before:
            try:
                page.goto(url_before, timeout=60000)
                page.wait_for_load_state("domcontentloaded", timeout=60000)
                page.wait_for_timeout(1500)
            except Exception:
                pass
    return count

def main():
    print("=" * 50)
    print("🤖 Facebook Multi-Page Scraper")
    print("=" * 50)

    links = load_links()
    print(f"📋 {len(links)} trang cần cào (tối đa {MAX_POSTS_PER_PAGE} bài/trang)")

    conn = init_db()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        load_cookies(context)

        if not check_logged_in(page):
            print("Vui lòng chạy lại: python login.py")
            browser.close()
            conn.close()
            return

        for idx, url in enumerate(links, 1):
            print("\n" + "#" * 50)
            print(f"# [{idx}/{len(links)}] {url}")
            print("#" * 50)
            try:
                scrape_page(page, conn, url)
            except Exception as e:
                print(f"❌ Lỗi khi cào trang {url}: {e}")

        input("\nNhấn Enter để đóng trình duyệt...")
        browser.close()

    conn.close()

main()