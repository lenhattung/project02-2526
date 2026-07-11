import { useEffect, useMemo, useState } from "react";
import { Bot, RefreshCw, Search } from "lucide-react";
import { ApiClient, LabelCode, NewsItem } from "../lib/api";

type Props = { api: ApiClient };

function labelTone(code?: LabelCode | null) {
  if (code === 0) return "bg-[#fff1eb] text-danger";
  if (code === 2) return "bg-[#edf8f2] text-success";
  if (code === 1) return "bg-slate-100 text-slate-700";
  return "bg-slate-100 text-slate-500";
}

function labelText(code?: LabelCode | null) {
  if (code === 0) return "0 - tiêu cực";
  if (code === 2) return "2 - tích cực";
  if (code === 1) return "1 - trung lập";
  return "Chưa label";
}

function LabelBadge({ code }: { code?: LabelCode | null }) {
  return <span className={`badge ${labelTone(code)}`}>{labelText(code)}</span>;
}

function compactText(text: string | null | undefined) {
  return text && text.trim() ? text : "Chưa có";
}

export function NewsPage({ api }: Props) {
  const [items, setItems] = useState<NewsItem[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [votedLabel, setVotedLabel] = useState("");
  const [selected, setSelected] = useState<NewsItem | null>(null);
  const [loading, setLoading] = useState(false);

  const query = useMemo(() => {
    const params = new URLSearchParams({ page_size: "20" });
    if (search) params.set("search", search);
    if (status) params.set("status", status);
    if (votedLabel) params.set("voted_label", votedLabel);
    return `?${params.toString()}`;
  }, [search, status, votedLabel]);

  async function load() {
    setLoading(true);
    try {
      const response = await api.news(query);
      setItems(response.items);
      setTotal(response.total);
      if (selected) {
        const refreshed = response.items.find((item) => item.id === selected.id);
        if (refreshed) setSelected(refreshed);
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(load, 250);
    return () => window.clearTimeout(timer);
  }, [query]);

  async function analyze(item: NewsItem) {
    await api.analyze(item.id);
    await load();
  }

  async function markDone(item: NewsItem) {
    await api.updateNews(item.id, { status: "processed" });
    await load();
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-3 md:flex-row md:items-end">
        <div>
          <h1 className="text-2xl font-bold">Quản lý tin/bài viết</h1>
          <p className="text-sm text-slate-500">{total} bản ghi đang có trong hệ thống.</p>
        </div>
        <button className="btn-quiet" onClick={load}>
          <RefreshCw size={16} />
          Làm mới
        </button>
      </div>
      <section className="panel p-4">
        <div className="grid gap-3 xl:grid-cols-[1fr_220px_220px]">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-2.5 text-slate-400" size={18} />
            <input
              className="h-10 w-full rounded-md border border-line pl-10 pr-3 outline-none focus:border-brand"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Tìm theo nội dung tin"
            />
          </div>
          <select className="h-10 rounded-md border border-line px-3 outline-none focus:border-brand" value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="">Tất cả trạng thái</option>
            <option value="new">Mới</option>
            <option value="pending">Chờ xử lý</option>
            <option value="needs_attention">Cần chú ý</option>
            <option value="processed">Đã xử lý</option>
          </select>
          <select className="h-10 rounded-md border border-line px-3 outline-none focus:border-brand" value={votedLabel} onChange={(event) => setVotedLabel(event.target.value)}>
            <option value="">Tất cả vote cảm xúc</option>
            <option value="0">0 - tiêu cực</option>
            <option value="1">1 - trung lập</option>
            <option value="2">2 - tích cực</option>
          </select>
        </div>
      </section>
      <section className="panel overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-[1260px] w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="w-[30%] px-4 py-3">Nội dung</th>
                <th className="px-4 py-3">Chủ đề</th>
                <th className="px-4 py-3">Mức độ</th>
                <th className="px-4 py-3">Gemini</th>
                <th className="px-4 py-3">SimCSE</th>
                <th className="px-4 py-3">PhoBERT</th>
                <th className="px-4 py-3">BGEM3</th>
                <th className="px-4 py-3">Vote</th>
                <th className="px-4 py-3">Trạng thái</th>
                <th className="px-4 py-3 text-right">Thao tác</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {loading ? (
                <tr>
                  <td className="px-4 py-8 text-center text-slate-500" colSpan={10}>
                    Đang tải dữ liệu...
                  </td>
                </tr>
              ) : null}
              {!loading && items.length === 0 ? (
                <tr>
                  <td className="px-4 py-8 text-center text-slate-500" colSpan={10}>
                    Chưa có dữ liệu phù hợp.
                  </td>
                </tr>
              ) : null}
              {items.map((item) => (
                <tr key={item.id} className="hover:bg-slate-50">
                  <td className="px-4 py-3">
                    <button className="line-clamp-2 text-left font-medium" onClick={() => setSelected(item)}>
                      {item.content}
                    </button>
                  </td>
                  <td className="px-4 py-3">
                    <span className="badge bg-slate-100 text-slate-700">{item.topic ?? "chưa phân loại"}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="badge bg-[#fff6e8] text-accent">{item.importance_level}</span>
                  </td>
                  <td className="px-4 py-3"><LabelBadge code={item.gemini_label} /></td>
                  <td className="px-4 py-3"><LabelBadge code={item.simcse_label} /></td>
                  <td className="px-4 py-3"><LabelBadge code={item.phobert_label} /></td>
                  <td className="px-4 py-3"><LabelBadge code={item.bgem3_label} /></td>
                  <td className="px-4 py-3"><LabelBadge code={item.voted_label} /></td>
                  <td className="px-4 py-3">
                    <span className="badge bg-[#edf8fa] text-brand">{item.label_status || item.status}</span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex justify-end gap-2">
                      <button className="btn-quiet h-9 px-2" onClick={() => analyze(item)} title="Phân tích AI">
                        <Bot size={16} />
                      </button>
                      <button className="btn-quiet h-9 px-2" onClick={() => markDone(item)}>Xử lý</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      {selected ? (
        <div className="fixed inset-0 z-30 bg-black/30 p-4" onClick={() => setSelected(null)}>
          <aside className="ml-auto h-full max-w-3xl overflow-auto rounded-md bg-white p-5 shadow-xl" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold">Chi tiết tin</h2>
                <p className="text-sm text-slate-500">ID #{selected.id}</p>
              </div>
              <button className="btn-quiet" onClick={() => setSelected(null)}>Đóng</button>
            </div>
            <p className="mt-5 whitespace-pre-wrap leading-7">{selected.content}</p>
            <div className="mt-5 grid gap-3 md:grid-cols-4">
              <div className="rounded-md border border-line p-3"><div className="text-xs text-slate-500">Chủ đề</div><div className="font-semibold">{selected.topic ?? "chưa phân loại"}</div></div>
              <div className="rounded-md border border-line p-3"><div className="text-xs text-slate-500">Mức độ</div><div className="font-semibold">{selected.importance_level}</div></div>
              <div className="rounded-md border border-line p-3"><div className="text-xs text-slate-500">Trạng thái tin</div><div className="font-semibold">{selected.status}</div></div>
              <div className="rounded-md border border-line p-3"><div className="text-xs text-slate-500">Trạng thái label</div><div className="font-semibold">{selected.label_status}</div></div>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-5">
              <div className="rounded-md border border-line p-3"><div className="text-xs text-slate-500">Gemini</div><div className="mt-2"><LabelBadge code={selected.gemini_label} /></div></div>
              <div className="rounded-md border border-line p-3"><div className="text-xs text-slate-500">SimCSE</div><div className="mt-2"><LabelBadge code={selected.simcse_label} /></div></div>
              <div className="rounded-md border border-line p-3"><div className="text-xs text-slate-500">PhoBERT</div><div className="mt-2"><LabelBadge code={selected.phobert_label} /></div></div>
              <div className="rounded-md border border-line p-3"><div className="text-xs text-slate-500">BGEM3</div><div className="mt-2"><LabelBadge code={selected.bgem3_label} /></div></div>
              <div className="rounded-md border border-line p-3"><div className="text-xs text-slate-500">Vote cuối</div><div className="mt-2"><LabelBadge code={selected.voted_label} /></div></div>
            </div>
            <div className="mt-4 rounded-md border border-line p-4">
              <div className="text-xs font-semibold uppercase text-slate-500">Lỗi model hoặc trạng thái thiếu</div>
              <div className="mt-2 text-sm text-slate-700 whitespace-pre-wrap">
                {compactText(selected.label_error_json ? JSON.stringify(selected.label_error_json, null, 2) : null)}
              </div>
            </div>
          </aside>
        </div>
      ) : null}
    </div>
  );
}
