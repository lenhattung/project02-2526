type Props = {
  label: string;
  value: number | string;
  helper: string;
  tone?: "brand" | "accent" | "success" | "danger";
};

const tones = {
  brand: "border-brand/30 bg-[#edf8fa]",
  accent: "border-accent/30 bg-[#fff6e8]",
  success: "border-success/30 bg-[#edf8f2]",
  danger: "border-danger/30 bg-[#fff1eb]",
};

export function MetricCard({ label, value, helper, tone = "brand" }: Props) {
  return (
    <div className={`panel border ${tones[tone]} p-4`}>
      <div className="text-sm font-semibold text-slate-600">{label}</div>
      <div className="mt-3 text-3xl font-bold text-ink">{value}</div>
      <div className="mt-2 text-xs text-slate-500">{helper}</div>
    </div>
  );
}
