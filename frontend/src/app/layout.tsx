import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "CPU-3DGS | Incremental 3D Gaussian Splatting Platform",
  description: "CPU-only video-to-3D Gaussian Splatting reconstruction & WebGL scene viewer",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#090d16] text-slate-100 antialiased">
        <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-[#090d16]/90 backdrop-blur">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3.5 sm:px-6">
            <Link href="/" className="flex items-center gap-2.5 font-semibold tracking-tight text-white">
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-sky-500/20 text-sky-400 ring-1 ring-sky-400/30">
                3D
              </span>
              <span>CPU-3DGS Studio</span>
              <span className="hidden rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-300 ring-1 ring-emerald-500/20 sm:inline-block">
                CPU-Only Engine
              </span>
            </Link>

            <nav className="flex items-center gap-2 sm:gap-5 text-sm font-medium text-slate-300">
              <Link href="/" className="rounded-md px-2.5 py-1.5 hover:bg-slate-800 hover:text-white">
                Overview
              </Link>
              <Link href="/upload" className="rounded-md px-2.5 py-1.5 hover:bg-slate-800 hover:text-white">
                New Reconstruction
              </Link>
              <Link
                href="/projects"
                className="rounded-md bg-sky-600 px-3 py-1.5 text-white hover:bg-sky-500"
              >
                Projects
              </Link>
            </nav>
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}

