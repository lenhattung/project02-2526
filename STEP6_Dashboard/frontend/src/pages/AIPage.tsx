import { useEffect, useState } from "react";
import { Bot, PlayCircle } from "lucide-react";
import { AIResult, ApiClient } from "../lib/api";

type Props = { api: ApiClient };

function sentimentText(result: AIResult) {
  if (result.sentiment_code === 0) return "0 - tiêu cực";
  if (result.sentiment_code === 2) return "2 - tích cực";
  return "1 - trung lập";
}

export function AIPage({ api }: Props) {
  const [results, setResults] = useState<AIResult[]>([]);
  const [loading, setLoading] = useState(false);

  async function load() {
    setResults(await api.aiResults());
  }

  async function runBatch() {
    setLoading(true);
    try {
      await api.analyzeBatch(50);
      await load();
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-3 md:flex-row md:items-end">
        <div>
          <h1 className="text-2xl font-bold">Phân tích AI</h1>
          <p className="text-sm text-slate-500">Gemini adapter giúp tóm tắt, phân loại, gán nhãn cảm xúc và gợi ý xử lý.</p>
        </div>
        <button className="btn-primary" onClick={runBatch} disabled={loading}>
          <PlayCircle size={18} />
          {loading ? "Đang phân tích..." : "Phân tích batch"}
        </button>
      </div>
      <section className="grid gap-4">
        {results.length === 0 ? (
          <div className="panel p-8 text-center text-slate-500">Chưa có kết quả AI. Hãy chạy phân tích batch hoặc phân tích từng tin ở trang quản lý.</div>
        ) : null}
        {results.map((result) => (
          <article key={result.id} className="panel p-4">
            <div className="flex flex-col justify-between gap-3 md:flex-row">
              <div className="flex items-center gap-3">
                <div className="grid h-10 w-10 place-items-center rounded-md bg-[#edf8fa] text-brand"><Bot size={18} /></div>
                <div>
                  <div className="font-bold">{result.category}</div>
                  <div className="text-xs text-slate-500">{result.provider} / {result.model_name}</div>
                </div>
              </div>
              <div className="flex gap-2">
                <span className="badge bg-slate-100 text-slate-700">{sentimentText(result)}</span>
                <span className="badge bg-[#fff6e8] text-accent">Score {Math.round(result.importance_score * 100)}</span>
                <span className={`badge ${result.attention_required ? "bg-[#fff1eb] text-danger" : "bg-[#edf8f2] text-success"}`}>{result.status_label}</span>
              </div>
            </div>
            <p className="mt-4 text-sm leading-6">{result.summary}</p>
            <div className="mt-4 rounded-md border border-line bg-slate-50 p-3 text-sm">
              <span className="font-semibold">Gợi ý xử lý: </span>{result.suggested_action}
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}
