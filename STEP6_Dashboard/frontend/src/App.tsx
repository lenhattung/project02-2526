import { useMemo, useState } from "react";
import { Layout, PageKey } from "./components/Layout";
import { ApiClient } from "./lib/api";
import { AIPage } from "./pages/AIPage";
import { LoginPage } from "./pages/LoginPage";
import { NewsPage } from "./pages/NewsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { ReportsPage } from "./pages/ReportsPage";
import { SettingsPage } from "./pages/SettingsPage";

function App() {
  const [token, setToken] = useState(localStorage.getItem("ctsv_access_token"));
  const [page, setPage] = useState<PageKey>("overview");
  const api = useMemo(() => new ApiClient(token), [token]);

  function handleLogin(nextToken: string) {
    localStorage.setItem("ctsv_access_token", nextToken);
    setToken(nextToken);
  }

  function logout() {
    localStorage.removeItem("ctsv_access_token");
    setToken(null);
  }

  if (!token) return <LoginPage onLogin={handleLogin} />;

  return (
    <Layout page={page} setPage={setPage} onLogout={logout}>
      {page === "overview" ? <OverviewPage api={api} /> : null}
      {page === "news" ? <NewsPage api={api} /> : null}
      {page === "ai" ? <AIPage api={api} /> : null}
      {page === "reports" ? <ReportsPage api={api} /> : null}
      {page === "settings" ? <SettingsPage api={api} /> : null}
    </Layout>
  );
}

export default App;
