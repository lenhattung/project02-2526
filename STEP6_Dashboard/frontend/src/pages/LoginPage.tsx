import { FormEvent, useState } from "react";
import { ShieldCheck } from "lucide-react";
import { ApiClient } from "../lib/api";

type Props = {
  onLogin: (token: string) => void;
};

export function LoginPage({ onLogin }: Props) {
  const [email, setEmail] = useState("admin@example.com");
  const [password, setPassword] = useState("Admin@123456");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await new ApiClient(null).login(email, password);
      onLogin(response.access_token);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không thể đăng nhập");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid min-h-screen place-items-center bg-[#eef4f8] p-4">
      <form onSubmit={submit} className="panel w-full max-w-md p-6">
        <div className="mb-6 flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-md bg-brand text-white">
            <ShieldCheck size={22} />
          </div>
          <div>
            <h1 className="text-xl font-bold">CTSV News Dashboard</h1>
            <p className="text-sm text-slate-500">Đăng nhập hệ thống quản lý tin/bài viết</p>
          </div>
        </div>
        <label className="text-sm font-semibold">Email</label>
        <input
          className="mt-2 h-11 w-full rounded-md border border-line px-3 outline-none focus:border-brand"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <label className="mt-4 block text-sm font-semibold">Mật khẩu</label>
        <input
          className="mt-2 h-11 w-full rounded-md border border-line px-3 outline-none focus:border-brand"
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        {error ? <div className="mt-4 rounded-md border border-danger/30 bg-[#fff1eb] p-3 text-sm text-danger">{error}</div> : null}
        <button className="btn-primary mt-6 w-full" disabled={loading}>
          {loading ? "Đang đăng nhập..." : "Đăng nhập"}
        </button>
      </form>
    </div>
  );
}
