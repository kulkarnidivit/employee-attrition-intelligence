import "./globals.css";
import Link from "next/link";

export const metadata = { title: "Employee Attrition Intelligence" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-50 text-slate-900">
        <nav className="border-b border-slate-200 bg-white">
          <div className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3 text-sm">
            <span className="font-semibold">Attrition Intelligence</span>
            <Link href="/" className="hover:underline">Overview</Link>
            <Link href="/drivers" className="hover:underline">Drivers</Link>
            <Link href="/risk" className="hover:underline">Risk center</Link>
          </div>
        </nav>
        <main className="mx-auto max-w-5xl space-y-4 px-4 py-6">
          {children}
          <p className="pt-4 text-xs text-slate-500">
            Decision support only. Scores and explanations describe the model, not causes. Synthetic demo data.
            Age, gender and marital status are not used by the model and are not shown.
          </p>
        </main>
      </body>
    </html>
  );
}
