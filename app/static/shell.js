// Shared app shell: the same collapsible sidebar on every page. Load in <head> (sets the theme before paint).
(function () {
  const LS = { get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } };
  document.documentElement.dataset.theme = LS.get("theme") === "dark" ? "dark" : "light";
  const ICON = {
    home: '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/>',
    ads: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9l5 3-5 3z"/>',
    record: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="currentColor"/>',
    results: '<path d="M3 17l5-6 4 3 4-7 5 9"/><path d="M3 21h18"/>',
  };
  const NAV = [["/home", "Home", "home"], ["/ads", "Ads", "ads"], ["/record", "Record", "record"], ["/results", "Results", "results"]];
  const alias = { "/": "/home", "/setup": "/record", "/analysis": "/results", "/review": "/ads" };
  const redraw = () => setTimeout(() => window.dispatchEvent(new Event("resize")), 200);
  function build() {
    const here = alias[location.pathname] || location.pathname;
    const side = document.createElement("aside");
    side.className = "side noprint";
    side.innerHTML = `<div class="brandmark"><span class="mk"><svg viewBox="0 0 24 24" fill="none" stroke="var(--accent-ink)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h3l2-6 4 12 3-9 2 3h6"/></svg></span>
        <span>Ad Copilot<small>EEG + AI ad analysis</small></span></div>
      <nav>${NAV.map(([href, name, ic]) => `<a href="${href}" class="${here === href ? "on" : ""}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICON[ic]}</svg>${name}</a>`).join("")}</nav>
      <div class="foot"><span class="led" id="shellLed"></span><span id="shellStatus">Ready</span><span style="flex-basis:100%"></span><button class="themebtn" id="themeBtn"></button></div>`;
    document.body.prepend(side);
    document.body.classList.add("has-side");
    const tg = document.createElement("button");
    tg.className = "sidetoggle noprint"; tg.title = "Show / hide the menu  ( [ )";
    tg.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></svg>';
    document.body.appendChild(tg);
    const setSide = off => { document.body.classList.toggle("side-off", off); LS.set("side", off ? "off" : "on"); redraw(); };
    if (LS.get("side") === "off") document.body.classList.add("side-off");
    tg.onclick = () => setSide(!document.body.classList.contains("side-off"));
    document.addEventListener("keydown", e => { if (e.key === "[" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) setSide(!document.body.classList.contains("side-off")); });
    const tb = document.getElementById("themeBtn");
    const label = () => tb.textContent = document.documentElement.dataset.theme === "dark" ? "Light theme" : "Dark theme";
    label();
    tb.onclick = () => { const d = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = d; LS.set("theme", d); label(); redraw(); window.dispatchEvent(new Event("themechange")); };
    status();
  }
  async function status() {
    try {
      const [run, proc] = await Promise.all([fetch("/api/session/status").then(r => r.json()), fetch("/api/process/status").then(r => r.json())]);
      const led = document.getElementById("shellLed"), t = document.getElementById("shellStatus");
      if (run.state === "running") { led.className = "led bad"; t.textContent = `Recording ${run.participant_id}`; }
      else if (proc.state === "running") { led.className = "led good"; t.textContent = `Analysing ${proc.participant_id}`; }
      else { led.className = "led"; t.textContent = "Ready"; }
    } catch (e) { /* server restarting */ }
    setTimeout(status, 5000);
  }
  if (document.body) build(); else document.addEventListener("DOMContentLoaded", build);
})();
