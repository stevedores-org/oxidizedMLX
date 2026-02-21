import { Link, useLocation } from "react-router-dom";

interface NavItem { label: string; href: string; external?: boolean; }
interface NavSection { title: string; items: NavItem[]; }

const DOCS = "https://docs.stevedores.org";
const navigation: NavSection[] = [
  { title: "Overview", items: [
    { label: "Introduction", href: "/" },
    { label: "Getting Started", href: "/getting-started" },
    { label: "Architecture", href: "/architecture" },
  ]},
  { title: "Crates", items: [
    { label: "mlx-core", href: "/crates/mlx-core" },
    { label: "mlx-ops", href: "/crates/mlx-ops" },
    { label: "mlx-cpu", href: "/crates/mlx-cpu" },
    { label: "mlx-metal", href: "/crates/mlx-metal" },
    { label: "mlx-nn", href: "/crates/mlx-nn" },
    { label: "mlx-io", href: "/crates/mlx-io" },
    { label: "mlx-sys", href: "/crates/mlx-sys" },
  ]},
  { title: "Ecosystem Docs", items: [
    { label: "llama.rs", href: `${DOCS}/llama-rs`, external: true },
    { label: "oxidizedRAG", href: `${DOCS}/oxidizedRAG`, external: true },
    { label: "oxidizedgraph", href: `${DOCS}/oxidizedgraph`, external: true },
  ]},
];

export default function Sidebar() {
  const { pathname } = useLocation();
  return (
    <nav className="w-64 shrink-0 bg-zinc-900/60 border-r border-zinc-800 fixed top-0 left-0 bottom-0 overflow-y-auto hidden lg:flex flex-col">
      <div className="px-5 pt-6 pb-4 border-b border-zinc-800/60">
        <Link to="/" className="flex items-center gap-2.5">
          <span className="text-orange-500 font-mono font-bold text-lg">⚡</span>
          <span className="font-bold text-lg tracking-tight text-zinc-100">oxidizedMLX</span>
        </Link>
      </div>
      <div className="flex-1 py-3">
        {navigation.map((s) => (
          <div key={s.title} className="mb-1">
            <div className="px-5 py-2 text-[11px] font-semibold text-zinc-500 uppercase tracking-widest">{s.title}</div>
            {s.items.map((item) => {
              const active = !item.external && pathname === item.href;
              const cls = `flex items-center gap-2 px-5 py-[7px] text-[13px] border-l-2 transition-all ${active ? "border-orange-500 text-orange-400 bg-orange-500/[0.06] font-medium" : "border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"}`;
              return item.external ? (
                <a key={item.href} href={item.href} className={cls}>{item.label}<span className="ml-auto text-[10px] text-zinc-600">&nearr;</span></a>
              ) : (
                <Link key={item.href} to={item.href} className={cls}>
                  {s.title === "Crates" && <span className="w-1.5 h-1.5 rounded-full bg-current opacity-50" />}
                  {item.label}
                </Link>
              );
            })}
          </div>
        ))}
      </div>
      <div className="px-5 py-4 border-t border-zinc-800/60 text-xs text-zinc-500 flex gap-3">
        <a href="https://github.com/stevedores-org/oxidizedMLX" className="hover:text-orange-400 transition">GitHub</a>
        <span className="text-zinc-700">&middot;</span>
        <a href="https://stevedores.org" className="hover:text-orange-400 transition">stevedores.org</a>
      </div>
    </nav>
  );
}
