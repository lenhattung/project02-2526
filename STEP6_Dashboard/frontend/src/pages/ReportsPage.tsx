import { useEffect, useState } from "react";
import { Download, FileSpreadsheet } from "lucide-react";
import { ApiClient } from "../lib/api";

type Props = { api: ApiClient };

export function ReportsPage({ api }: Props) {
  const [format, setFormat] = useState("csv");
  const [type, setType] = useState("monthly");
  const [reports, setReports] = useState<Array<{ id: number; report_type: string; format: string; status: string; created_at: string }>>([]);
  const [loading, setLoading] = useState(false);

  async function load() {
    setReports(await api.reports());
  }

  async function create() {
    setLoading(true);
    try {
      await api.createReport(format, type);
      await load();
    } finally {
      setLoading(false);
    }
  }

  async function download(id: number) {
    const blob = await api.downloadReport(id);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `ctsv_report_${id}`;
    link.click();
    URL.revokeObjectURL(url);
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold">Báo cáo</h1>
        <p className="text-sm text-slate-500">Tạo và tải báo cáo phục vụ tổng hợp tình hình công tác sinh viên.</p>
      </div>
      <section className="panel p-4">
        <div className="grid gap-3 md:grid-cols-[220px_220px_auto]">
          <select className="h-10 rounded-md border border-line px-3" value={type} onChange={(event) => setType(event.target.value)}>
            <option value="daily">Ngày</option>
            <option value="weekly">Tuần</option>
            <option value="monthly">Tháng</option>
            <option value="yearly">Năm</option>
          </select>
          <select className="h-10 rounded-md border border-line px-3" value={format} onChange={(event) => setFormat(event.target.value)}>
            <option value="csv">CSV</option>
            <option value="xlsx">Excel</option>
            <option value="pdf">PDF</option>
          </select>
          <button className="btn-primary justify-self-start" onClick={create} disabled={loading}>
            <FileSpreadsheet size={18} />
            {loading ? "Đang tạo..." : "Tạo báo cáo"}
          </button>
        </div>
      </section>
      <section className="panel overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs uppercase text-slate-500">
            <tr><th className="px-4 py-3">ID</th><th className="px-4 py-3">Loại</th><th className="px-4 py-3">Định dạng</th><th className="px-4 py-3">Trạng thái</th><th className="px-4 py-3 text-right">Tải</th></tr>
          </thead>
          <tbody className="divide-y divide-line">
            {reports.map((report) => (
              <tr key={report.id}>
                <td className="px-4 py-3">#{report.id}</td>
                <td className="px-4 py-3">{report.report_type}</td>
                <td className="px-4 py-3">{report.format}</td>
                <td className="px-4 py-3"><span className="badge bg-[#edf8f2] text-success">{report.status}</span></td>
                <td className="px-4 py-3 text-right">
                  <button className="btn-quiet" onClick={() => download(report.id)}>
                    <Download size={16} />
                    Tải xuống
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
