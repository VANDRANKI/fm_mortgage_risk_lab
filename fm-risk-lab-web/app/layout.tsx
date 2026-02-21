import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import StarField from "@/components/StarField";

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Mortgage Credit Risk & Stress Testing Lab",
  description:
    "Production-grade ECL computation, PD/LGD modeling, and stress-scenario analysis for Freddie Mac mortgage portfolios.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-[#0a0f1e] text-gray-100 font-mono">
        <StarField />
        <Navbar />
        <main className="relative z-[1] max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
