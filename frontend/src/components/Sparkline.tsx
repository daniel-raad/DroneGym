import React from "react";

type Props = {
  values: number[];
  width?: number;
  height?: number;
  stroke?: string;
  fill?: string;
  yMin?: number;
  yMax?: number;
};

export function Sparkline({
  values,
  width = 220,
  height = 40,
  stroke = "#5db8ff",
  fill = "rgba(93,184,255,0.18)",
  yMin,
  yMax,
}: Props) {
  if (!values.length) {
    return (
      <div
        style={{
          height,
          width,
          color: "var(--muted)",
          fontSize: 11,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        no data
      </div>
    );
  }
  const lo = yMin ?? Math.min(...values);
  const hi = yMax ?? Math.max(...values);
  const span = hi - lo || 1;
  const stepX = width / Math.max(values.length - 1, 1);
  const pts = values.map((v, i) => {
    const x = i * stepX;
    const y = height - ((v - lo) / span) * (height - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const path = `M ${pts[0]} L ${pts.slice(1).join(" L ")}`;
  const area = `${path} L ${(values.length - 1) * stepX},${height} L 0,${height} Z`;
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <path d={area} fill={fill} />
      <path d={path} fill="none" stroke={stroke} strokeWidth={1.5} />
      <text
        x={2}
        y={10}
        fontSize={9}
        fill="var(--muted)"
        fontFamily="ui-monospace, monospace"
      >
        {hi.toFixed(2)}
      </text>
      <text
        x={2}
        y={height - 2}
        fontSize={9}
        fill="var(--muted)"
        fontFamily="ui-monospace, monospace"
      >
        {lo.toFixed(2)}
      </text>
    </svg>
  );
}
