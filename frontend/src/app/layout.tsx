import type { Metadata } from "next";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import "@/styles/globals.css";

export const metadata: Metadata = {
  title: "BioDreamer — World Models for Biological Design",
  description:
    "Teaching machines to dream about biology. A unified framework for model-based RL applied to molecular dynamics, protein engineering, and cell reprogramming.",
  keywords: [
    "BioDreamer",
    "world models",
    "protein design",
    "model-based RL",
    "active inference",
    "molecular dynamics",
    "cell reprogramming",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
