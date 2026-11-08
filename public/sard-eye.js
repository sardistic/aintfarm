// The sardistic eye: a link home whose iris follows the pointer, with a label that slides out on hover/focus.
(() => {
  if (document.querySelector('.sard-eye')) return;
  const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const style = document.createElement('style');
  style.textContent = `
    .sard-eye { position: fixed; left: 10px; bottom: 10px; z-index: 50; display: flex; align-items: center; gap: 8px;
      height: 32px; text-decoration: none; color: #f4ead6; -webkit-tap-highlight-color: transparent; }
    .sard-eye svg { width: 32px; height: 32px; flex: none; filter: drop-shadow(0 0 4px rgba(0,0,0,.85));
      opacity: .78; transition: opacity .18s, transform .18s; }
    .sard-eye:hover svg, .sard-eye:focus-visible svg { opacity: 1; transform: scale(1.12); }
    .sard-eye span { font: 600 12px/1 ui-monospace, Consolas, monospace; letter-spacing: .06em; white-space: nowrap;
      padding: 6px 10px; border-radius: 999px; background: rgba(12, 10, 14, .82); border: 1px solid rgba(247, 207, 90, .35);
      box-shadow: 0 4px 14px rgba(0,0,0,.5); backdrop-filter: blur(6px);
      opacity: 0; transform: translateX(-8px) scale(.92); transform-origin: left center; pointer-events: none;
      transition: opacity .18s ease, transform .22s cubic-bezier(.2, 1.4, .4, 1); }
    .sard-eye:hover span, .sard-eye:focus-visible span { opacity: 1; transform: none; }
    .sard-eye:focus-visible { outline: none; }
    .sard-eye:focus-visible svg { outline: 2px solid #f7cf5a; outline-offset: 2px; border-radius: 50%; }
  `;
  document.head.appendChild(style);

  const a = document.createElement('a');
  a.className = 'sard-eye';
  a.href = 'https://sardistic.com/';
  a.setAttribute('aria-label', 'sardistic.com');
  a.innerHTML = `
    <svg viewBox="0 0 64 64" aria-hidden="true">
      <defs>
        <radialGradient id="sgsEye-lens" cx="40%" cy="34%" r="70%"><stop offset="0" stop-color="#fbf8f2"/><stop offset=".7" stop-color="#ebe5da"/><stop offset="1" stop-color="#d9d2c4"/></radialGradient>
        <radialGradient id="sgsEye-amber" cx="42%" cy="38%" r="62%"><stop offset="0" stop-color="#f7cf5a"/><stop offset="1" stop-color="#d6921a"/></radialGradient>
        <radialGradient id="sgsEye-red" cx="40%" cy="36%" r="64%"><stop offset="0" stop-color="#f06a52"/><stop offset="1" stop-color="#ad2416"/></radialGradient>
        <radialGradient id="sgsEye-glint" cx="31%" cy="26%" r="48%"><stop offset="0" stop-color="#fff" stop-opacity=".85"/><stop offset=".35" stop-color="#fff" stop-opacity=".22"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
        <linearGradient id="sgsEye-rim" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff"/><stop offset=".5" stop-color="#fff" stop-opacity=".35"/><stop offset="1" stop-color="#fff" stop-opacity=".7"/></linearGradient>
        <clipPath id="sgsEye-clip"><circle cx="32" cy="32" r="29.5"/></clipPath>
      </defs>
      <circle cx="32" cy="32" r="30" fill="url(#sgsEye-lens)"/>
      <g clip-path="url(#sgsEye-clip)"><g class="sgs-iris">
        <circle cx="32" cy="32" r="18.5" fill="url(#sgsEye-amber)"/>
        <circle cx="32" cy="32" r="18.5" fill="none" stroke="#fff" stroke-opacity=".55" stroke-width="1"/>
        <g class="sgs-pupil">
          <circle cx="32" cy="32" r="9" fill="url(#sgsEye-red)"/>
          <circle cx="32" cy="32" r="9" fill="none" stroke="#fff" stroke-opacity=".55" stroke-width=".9"/>
        </g>
      </g></g>
      <circle cx="32" cy="32" r="30" fill="url(#sgsEye-glint)"/>
      <circle cx="32" cy="32" r="29.3" fill="none" stroke="url(#sgsEye-rim)" stroke-width="1.4"/>
      <circle cx="32" cy="32" r="30.6" fill="none" stroke="#16141b" stroke-opacity=".28" stroke-width="1.2"/>
    </svg>
    <span>sardistic.com</span>`;
  const mount = () => document.body.appendChild(a);
  if (document.body) mount(); else addEventListener('DOMContentLoaded', mount);

  if (REDUCE) return;
  const svg = a.querySelector('svg'), iris = a.querySelector('.sgs-iris'), pupil = a.querySelector('.sgs-pupil');
  let tx = 0, ty = 0, x = 0, y = 0, raf = 0;
  const MAX = 8.5;                       // how far (in the 64-unit viewBox) the iris can travel from centre
  function aim(px, py) {
    const r = svg.getBoundingClientRect();
    const dx = px - (r.left + r.width / 2), dy = py - (r.top + r.height / 2);
    const d = Math.hypot(dx, dy) || 1;
    const reach = MAX * Math.min(1, d / 160);   // near the eye it looks less far, so it doesn't snap
    tx = dx / d * reach; ty = dy / d * reach;
    if (!raf) raf = requestAnimationFrame(tick);
  }
  function tick() {
    x += (tx - x) * 0.22; y += (ty - y) * 0.22;
    iris.setAttribute('transform', `translate(${x.toFixed(2)} ${y.toFixed(2)})`);
    pupil.setAttribute('transform', `translate(${(x * 0.35).toFixed(2)} ${(y * 0.35).toFixed(2)})`);  // a little parallax
    raf = Math.abs(tx - x) + Math.abs(ty - y) > 0.02 ? requestAnimationFrame(tick) : 0;
  }
  let lastPointer = -1e9;
  const fromPointer = e => { lastPointer = performance.now(); aim(e.clientX, e.clientY); };
  addEventListener('pointermove', fromPointer, { passive: true });
  addEventListener('pointerdown', fromPointer, { passive: true });
  document.addEventListener('pointerleave', () => { tx = 0; ty = 0; if (!raf) raf = requestAnimationFrame(tick); });

  // Tilt: on phones and tablets the iris rolls toward whichever edge is tipped down, like a marble.
  // Uses device orientation (accelerometer fused with the gyroscope). The way you normally hold the
  // device becomes "centre" and slowly re-centres; the pointer takes over again for 2 s after it moves.
  const TILT_RANGE = 25;                 // degrees of tilt for the iris to reach the rim
  let baseBeta = null, baseGamma = null, lastTilt = 0;
  function onTilt(e) {
    if (e.beta == null || e.gamma == null) return;
    const now = performance.now();
    if (baseBeta === null) { baseBeta = e.beta; baseGamma = e.gamma; }
    const k = Math.min(1, (now - (lastTilt || now)) / 4000);   // ~4 s drift back to the resting angle
    baseBeta += (e.beta - baseBeta) * k; baseGamma += (e.gamma - baseGamma) * k;
    lastTilt = now;
    if (now - lastPointer < 2000) return;
    // device frame: gamma > 0 tips the right edge down, beta rising tips the bottom edge down
    const vx = (e.gamma - baseGamma) / TILT_RANGE, vy = (e.beta - baseBeta) / TILT_RANGE;
    const a = ((screen.orientation && screen.orientation.angle) || window.orientation || 0) * Math.PI / 180;
    let sx = vx * Math.cos(a) + vy * Math.sin(a), sy = -vx * Math.sin(a) + vy * Math.cos(a);
    const m = Math.hypot(sx, sy);
    if (m > 1) { sx /= m; sy /= m; }
    tx = sx * MAX; ty = sy * MAX;
    if (!raf) raf = requestAnimationFrame(tick);
  }
  const DOE = window.DeviceOrientationEvent;
  if (DOE) {
    addEventListener('deviceorientation', onTilt, { passive: true });
    // iOS Safari only starts delivering events after a permission prompt, which must follow a tap.
    // (Some other browsers expose requestPermission too and simply resolve 'granted'.)
    if (typeof DOE.requestPermission === 'function') {
      addEventListener('touchend', () => { DOE.requestPermission().catch(() => {}); }, { once: true, passive: true });
    }
  }
})();
