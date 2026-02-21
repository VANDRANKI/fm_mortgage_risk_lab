"use client";

interface SliderProps {
  label:    string;
  unit:     string;
  min:      number;
  max:      number;
  step:     number;
  value:    number;
  onChange: (v: number) => void;
  color?:   string;
}

export default function ScenarioSlider({
  label, unit, min, max, step, value, onChange, color = "#22d3ee",
}: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-baseline">
        <span className="text-xs text-gray-400 uppercase tracking-wide">{label}</span>
        <span
          className="text-base font-bold tabular-nums"
          style={{ color }}
        >
          {value >= 0 ? "+" : ""}{value.toFixed(1)}{unit}
        </span>
      </div>
      <div className="relative h-2 bg-gray-800 rounded-full">
        {/* Filled track */}
        <div
          className="absolute h-full rounded-full transition-all"
          style={{ width: `${pct}%`, background: color }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="absolute inset-0 w-full opacity-0 cursor-pointer h-full"
        />
      </div>
      <div className="flex justify-between text-xs text-gray-600">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}
