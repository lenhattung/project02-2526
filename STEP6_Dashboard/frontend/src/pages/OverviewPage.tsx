import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle } from "lucide-react";
import { ApiClient, LabelCode, NewsItem, Summary } from "../lib/api";
import { MetricCard } from "../components/MetricCard";

type Props = { api: ApiClient };

function labelText(code?: LabelCode | null) {
  if (code === 0) return "0 - tiêu cực";
  if (code === 2) return "2 - tích cực";
  if (code === 1) return "1 - trung lập";
  return "Chưa label";
}

function labelColor(code?: string) {
  if (code === "0") return "#d95f43";
  if (code === "2") return "#2f855a";
  if (code === "1") return "#1f7a8c";
  return "#94a3b8";
}

export function OverviewPage({ api }: Props) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [trends, setTrends] = useState<Array<{ date: string; count: number }>>([]);
  const [topics, setTopics] = useState<Array<{ name: string; value: number }>>([]);
  const [importance, setImportance] = useState<Array<{ name: string; value: number }>>([]);
  const [sentiments, setSentiments] = useState<Array<{ name: string; value: number }>>([]);
  const [latest, setLatest] = useState<NewsItem[]>([]);
  const [alerts, setAlerts] = useState<NewsItem[]>([]);

  useEffect(() => {
    Promise.all([api.summary(), api.trends(), api.distributions(), api.news("?page_size=6"), api.alerts()]).then(([s, t, d, n, a]) => {
      setSummary(s);
      setTrends(t);
      setTopics(d.topics);
      setImportance(d.importance);
      setSentiments(d.sentiments);
      setLatest(n.items);
      setAlerts(a);
    });
  }, [api]);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold">Tổng quan vận hành</h1>
        <p className="text-sm text-slate-500">Theo dõi dữ liệu thu thập, cảnh báo và kết quả phân tích AI.</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricCard label="Tổng tin" value={summary?.total ?? "..."} helper="Đã lưu online" />
        <MetricCard label="Tin mới hôm nay" value={summary?.today ?? "..."} helper="Theo thời điểm ingest" tone="success" />
        <MetricCard label="Tin quan trọng" value={summary?.important ?? "..."} helper="High hoặc critical" tone="danger" />
        <MetricCard label="Chưa xử lý" value={summary?.pending ?? "..."} helper="Trạng thái new/pending" tone="accent" />
        <MetricCard label="Đã AI" value={summary?.analyzed ?? "..."} helper="Có vote cảm xúc" />
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <section className="panel p-4">
          <h2 className="font-bold">Xu hướng tin theo thời gian</h2>
          <div className="mt-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={12} />
                <YAxis allowDecimals={false} fontSize={12} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#1f7a8c" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="panel p-4">
          <h2 className="font-bold">Tin cần chú ý</h2>
          <div className="mt-4 space-y-3">
            {alerts.length === 0 ? <div className="text-sm text-slate-500">Chưa có cảnh báo mức cao.</div> : null}
            {alerts.map((item) => (
              <div key={item.id} className="rounded-md border border-danger/20 bg-[#fff7f2] p-3">
                <div className="flex items-center gap-2 text-sm font-bold text-danger">
                  <AlertTriangle size={16} />
                  {item.importance_level}
                </div>
                <p className="mt-2 line-clamp-2 text-sm">{item.content}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
      <div className="grid gap-4 xl:grid-cols-3">
        <section className="panel p-4">
          <h2 className="font-bold">Phân loại chủ đề</h2>
          <div className="mt-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={topics} dataKey="value" nameKey="name" outerRadius={90} label>
                  {topics.map((_, index) => <Cell key={index} fill={["#1f7a8c", "#bf6f13", "#2f855a", "#cc5c2c"][index % 4]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="panel p-4">
          <h2 className="font-bold">Mức độ quan trọng</h2>
          <div className="mt-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" fill="#bf6f13" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="panel p-4">
          <h2 className="font-bold">Vote cảm xúc</h2>
          <div className="mt-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sentiments}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {sentiments.map((entry, index) => <Cell key={index} fill={labelColor(entry.name)} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
      <section className="panel overflow-hidden">
        <div className="border-b border-line p-4 font-bold">Tin mới nhất</div>
        <div className="divide-y divide-line">
          {latest.map((item) => (
            <div key={item.id} className="grid gap-2 p-4 md:grid-cols-[1fr_180px_160px_160px]">
              <div className="line-clamp-2 text-sm">{item.content}</div>
              <span className="badge bg-slate-100 text-slate-700">{item.topic ?? "chưa phân loại"}</span>
              <span className="badge bg-slate-100 text-slate-700">{labelText(item.voted_label)}</span>
              <span className="badge bg-[#edf8fa] text-brand">{item.label_status || item.status}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
