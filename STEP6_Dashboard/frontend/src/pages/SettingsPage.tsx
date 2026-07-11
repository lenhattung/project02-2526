import { useEffect, useState } from "react";
import { KeyRound, Plus } from "lucide-react";
import { ApiClient, Source, TokenRow } from "../lib/api";

type Props = { api: ApiClient };

export function SettingsPage({ api }: Props) {
  const [sources, setSources] = useState<Source[]>([]);
  const [tokens, setTokens] = useState<TokenRow[]>([]);
  const [tokenName, setTokenName] = useState("Desktop CTSV");
  const [newToken, setNewToken] = useState("");

  async function load() {
    const [sourceRows, tokenRows] = await Promise.all([api.sources(), api.tokens()]);
    setSources(sourceRows);
    setTokens(tokenRows);
  }

  async function createToken() {
    const result = await api.createToken(tokenName);
    setNewToken(result.token);
    await load();
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold">Cấu hình hệ thống</h1>
        <p className="text-sm text-slate-500">Quản lý nguồn dữ liệu, API token Desktop Tool và cấu hình vận hành.</p>
      </div>
      <section className="panel p-4">
        <h2 className="font-bold">API token cho Desktop Tool</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-[1fr_auto]">
          <input className="h-10 rounded-md border border-line px-3 outline-none focus:border-brand" value={tokenName} onChange={(event) => setTokenName(event.target.value)} />
          <button className="btn-primary" onClick={createToken}>
            <Plus size={18} />
            Tạo token
          </button>
        </div>
        {newToken ? (
          <div className="mt-4 rounded-md border border-success/30 bg-[#edf8f2] p-3 text-sm">
            <div className="font-semibold text-success">Token mới chỉ hiển thị một lần</div>
            <code className="mt-2 block break-all rounded bg-white p-2">{newToken}</code>
          </div>
        ) : null}
        <div className="mt-4 divide-y divide-line">
          {tokens.map((token) => (
            <div key={token.id} className="flex items-center justify-between py-3">
              <div className="flex items-center gap-3">
                <KeyRound size={18} className="text-brand" />
                <div><div className="font-semibold">{token.name}</div><div className="text-xs text-slate-500">{token.scope}</div></div>
              </div>
              <span className={`badge ${token.is_active ? "bg-[#edf8f2] text-success" : "bg-slate-100 text-slate-500"}`}>{token.is_active ? "active" : "inactive"}</span>
            </div>
          ))}
        </div>
      </section>
      <section className="panel p-4">
        <h2 className="font-bold">Nguồn dữ liệu</h2>
        <div className="mt-4 grid gap-3">
          {sources.map((source) => (
            <div key={source.id} className="rounded-md border border-line p-3">
              <div className="font-semibold">{source.name}</div>
              <div className="text-sm text-slate-500">{source.url}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
