export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8010/api";

export type Summary = {
  total: number;
  today: number;
  important: number;
  pending: number;
  analyzed: number;
};

export type LabelCode = 0 | 1 | 2;

export type NewsItem = {
  id: number;
  content: string;
  status: string;
  topic?: string | null;
  importance_level: string;
  collected_at?: string | null;
  created_at: string;
  like_count?: number | null;
  comment_count?: number | null;
  gemini_label?: LabelCode | null;
  simcse_label?: LabelCode | null;
  phobert_label?: LabelCode | null;
  bgem3_label?: LabelCode | null;
  voted_label?: LabelCode | null;
  label_status: string;
  labeled_at?: string | null;
  label_error_json?: Record<string, string> | null;
  ai_results?: AIResult[];
};

export type AIResult = {
  id: number;
  summary: string;
  category: string;
  importance_score: number;
  attention_required: boolean;
  suggested_action: string;
  sentiment_label: string;
  sentiment_code: number;
  status_label: string;
  provider: string;
  model_name: string;
  created_at: string;
};

export type Source = {
  id: number;
  name: string;
  url: string;
  platform: string;
  source_type: string;
  is_active: boolean;
};

export type TokenRow = {
  id: number;
  name: string;
  scope: string;
  is_active: boolean;
  last_used_at?: string | null;
  created_at: string;
};

export class ApiClient {
  token: string | null;

  constructor(token: string | null) {
    this.token = token;
  }

  async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const headers = new Headers(options.headers);
    headers.set("Content-Type", "application/json");
    if (this.token) headers.set("Authorization", `Bearer ${this.token}`);
    let response: Response;
    try {
      response = await fetch(`${API_BASE}${path}`, { ...options, headers });
    } catch (error) {
      if (error instanceof TypeError) {
        throw new Error(`Không kết nối được backend tại ${API_BASE}. Hãy kiểm tra backend đã chạy ở cổng 8010 chưa.`);
      }
      throw error;
    }
    if (!response.ok) {
      const raw = await response.text();
      let message = raw || `API error ${response.status}`;
      try {
        const parsed = JSON.parse(raw) as { detail?: string };
        if (parsed.detail) {
          message = parsed.detail;
        }
      } catch {
        // keep raw response text when it is not JSON
      }
      if (response.status === 404) {
        throw new Error(`Không tìm thấy API ${path}. Hãy kiểm tra backend đang chạy đúng URL ${API_BASE}.`);
      }
      throw new Error(message);
    }
    if (response.status === 204) return undefined as T;
    return response.json() as Promise<T>;
  }

  login(email: string, password: string) {
    return this.request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  }

  summary() {
    return this.request<Summary>("/dashboard/summary");
  }

  trends() {
    return this.request<Array<{ date: string; count: number }>>("/dashboard/trends");
  }

  distributions() {
    return this.request<{ topics: Array<{ name: string; value: number }>; importance: Array<{ name: string; value: number }>; sentiments: Array<{ name: string; value: number }> }>("/dashboard/distributions");
  }

  alerts() {
    return this.request<NewsItem[]>("/dashboard/alerts");
  }

  news(params = "") {
    return this.request<{ items: NewsItem[]; total: number; page: number; page_size: number }>(`/news${params}`);
  }

  updateNews(id: number, payload: Partial<NewsItem>) {
    return this.request<NewsItem>(`/news/${id}`, { method: "PATCH", body: JSON.stringify(payload) });
  }

  analyze(id: number) {
    return this.request<AIResult>(`/ai/analyze/${id}`, { method: "POST" });
  }

  analyzeBatch(limit = 50) {
    return this.request<{ analyzed: number }>(`/ai/analyze-batch?limit=${limit}`, { method: "POST" });
  }

  aiResults() {
    return this.request<AIResult[]>("/ai/results");
  }

  createReport(format: string, report_type: string) {
    return this.request<{ id: number; file_path: string; status: string }>("/reports", {
      method: "POST",
      body: JSON.stringify({ format, report_type }),
    });
  }

  reports() {
    return this.request<Array<{ id: number; report_type: string; format: string; status: string; created_at: string }>>("/reports");
  }

  async downloadReport(id: number) {
    const headers = new Headers();
    if (this.token) headers.set("Authorization", `Bearer ${this.token}`);
    const response = await fetch(`${API_BASE}/reports/${id}/download`, { headers });
    if (!response.ok) throw new Error(await response.text());
    return response.blob();
  }

  sources() {
    return this.request<Source[]>("/sources");
  }

  tokens() {
    return this.request<TokenRow[]>("/tokens");
  }

  createToken(name: string) {
    return this.request<TokenRow & { token: string }>("/tokens", { method: "POST", body: JSON.stringify({ name }) });
  }
}
