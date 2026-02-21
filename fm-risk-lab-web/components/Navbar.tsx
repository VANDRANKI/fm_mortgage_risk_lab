"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/",              label: "Portfolio" },
  { href: "/risk-lab",     label: "Risk Lab" },
  { href: "/loan-explorer",label: "Loan Explorer" },
  { href: "/methodology",  label: "Methodology" },
];

export default function Navbar() {
  const path = usePathname();
  return (
    <nav className="border-b border-[#1e3a5f] bg-[#080d1a]/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-14">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2 group">
          <span className="text-[#22d3ee] text-lg font-bold tracking-tight group-hover:opacity-80 transition-opacity">
            FM<span className="text-white">Risk</span>Lab
          </span>
          <span className="text-xs text-gray-500 hidden sm:block">Mortgage Credit Risk</span>
        </Link>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          {LINKS.map(({ href, label }) => {
            const active = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`
                  px-3 py-1.5 rounded-md text-xs transition-all
                  ${active
                    ? "bg-[#22d3ee]/10 text-[#22d3ee] border border-[#22d3ee]/30"
                    : "text-gray-400 hover:text-gray-200 hover:bg-white/5"
                  }
                `}
              >
                {label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
