import { Bell, Bot, FileBarChart, LayoutDashboard, LogOut, Newspaper, Search, Settings } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import dntuLogo from "../asset/image/Logo-DH-Cong-Nghe-Dong-Nai-DNTU.webp";

export type PageKey = "overview" | "news" | "ai" | "reports" | "settings";

const navItems: Array<{ key: PageKey; label: string; icon: LucideIcon }> = [
  { key: "overview", label: "Tổng quan", icon: LayoutDashboard },
  { key: "news", label: "Quản lý tin", icon: Newspaper },
  { key: "ai", label: "Phân tích AI", icon: Bot },
  { key: "reports", label: "Báo cáo", icon: FileBarChart },
  { key: "settings", label: "Cấu hình", icon: Settings },
];

type Props = {
  page: PageKey;
  setPage: (page: PageKey) => void;
  onLogout: () => void;
  children: ReactNode;
};

export function Layout({ page, setPage, onLogout, children }: Props) {
  return (
    <div className="min-h-screen bg-[#f4f7fb] text-ink">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-64 border-r border-line bg-white lg:block">
        <div className="flex h-16 items-center gap-3 border-b border-line px-5">
          <div className="grid h-12 w-12 place-items-center overflow-hidden rounded-md border border-line bg-white">
            <img src={dntuLogo} alt="Logo DNTU" className="h-10 w-10 object-contain" />
          </div>
          <div>
            <div className="text-sm font-bold">Quản lý cảm xúc Sinh Viên</div>
            <div className="text-xs text-slate-500">News Operations</div>
          </div>
        </div>
        <nav className="space-y-1 p-3">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = page === item.key;
            return (
              <button
                key={item.key}
                onClick={() => setPage(item.key)}
                className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm font-semibold ${
                  active ? "bg-[#e8f4f6] text-brand" : "text-slate-600 hover:bg-slate-50"
                }`}
              >
                <Icon size={18} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </aside>
      <div className="lg:pl-64">
        <header className="sticky top-0 z-10 flex h-16 items-center justify-between border-b border-line bg-white px-4 lg:px-6">
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <div className="relative w-full max-w-lg">
              <Search className="pointer-events-none absolute left-3 top-2.5 text-slate-400" size={18} />
              <input className="h-10 w-full rounded-md border border-line bg-slate-50 pl-10 pr-3 text-sm outline-none focus:border-brand" placeholder="Tìm kiếm nhanh theo nội dung, chủ đề, trạng thái" />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="btn-quiet h-10 w-10 px-0" title="Thông báo">
              <Bell size={18} />
            </button>
            <button className="btn-quiet" onClick={onLogout}>
              <LogOut size={18} />
              Đăng xuất
            </button>
          </div>
        </header>
        <main className="p-4 lg:p-6">{children}</main>
      </div>
    </div>
  );
}
