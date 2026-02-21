"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from "recharts";
import type { VintageSeries } from "@/lib/api";

const COLORS = [
  "#22d3ee", "#f59e0b", "#a78bfa", "#34d399",
  "#f472b6", "#60a5fa", "#fb923c",
];

interface Props {
  data: VintageSeries[];
}

export default function VintageChart({ data }: Props) {
  // Pivot: obs_year → columns per vintage
  const years = Array.from(
    new Set(data.flatMap((s) => s.data.map((d) => d.obs_year)))
  ).sort();

  const pivoted = years.map((yr) => {
    const row: Record<string, number | string> = { year: yr };
    data.forEach((s) => {
      const pt = s.data.find((d) => d.obs_year === yr);
      if (pt) row[`${s.vintage_year}`] = +(pt.default_rate * 100).toFixed(3);
    });
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={pivoted} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
        <XAxis dataKey="year" stroke="#475569" tick={{ fontSize: 11 }} />
        <YAxis
          tickFormatter={(v) => `${v}%`}
          stroke="#475569"
          tick={{ fontSize: 11 }}
          width={48}
        />
        <Tooltip
          contentStyle={{ background: "#0f172a", border: "1px solid #1e3a5f", fontSize: 12 }}
          formatter={(v: number | undefined) => [`${v ?? 0}%`, ""]}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {data.map((s, i) => (
          <Line
            key={s.vintage_year}
            dataKey={`${s.vintage_year}`}
            name={`${s.vintage_year} vintage`}
            stroke={COLORS[i % COLORS.length]}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
