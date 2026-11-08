// Field Life — every particle is a unique, untrained, mutating neural net with mutable physics.
// Particles never interact directly: they read, absorb from and write to a G×G×(F+1) field tensor
// (F evolvable channels + a density channel). All simulation state lives on the GPU (WebGPU compute).
// A CPU-side gene bank archives evolved genomes as phenotype vectors (nearest-neighbour niches).

const qs = new URLSearchParams(location.search);
const F = clampInt(qs.get('f'), 4, 16, 8);          // evolvable field channels (ch0 = energy)
const HMAX = clampInt(qs.get('h'), 2, 32, 12);      // max hidden width (actual width is a gene)
const G = clampInt(qs.get('g'), 128, 2048, 768);    // field grid size
const N_REQ = clampInt(qs.get('n'), 1024, 2_000_000, 262144);
const SOURCES = 12;
const FS = F + 1;                // field stride: F channels + density
const KE = 4;                    // effect-gene slots per particle
const EFFECTS = ['none', 'taxis', 'swirl', 'isoline', 'catalyse', 'jet', 'anchor', 'pressure', 'pulse', 'siphon', 'relay', 'heat', 'shield', 'drift'];
const NT = EFFECTS.length;
const NI = 3 * FS + 6;           // value+gradx+grady per channel (incl. density), energy, vel.xy, mem.xy, clock
const NO = 6 + F + KE;           // split, share, write gate, write amt, aim[F], mem[2], effect gates[KE]
const P = NI * HMAX + HMAX + HMAX * NO + NO;
const HDR = 8;                   // H, range, mutRate, hue, lineage, generation, maxAge, clockFreq
const EOFF = HDR + 3 * F;        // chemistry: absorb[F], energy gain[F], push[F]
const WOFF = EOFF + 4 * KE;      // effects: (type|chanA<<8|chanB<<16, p0, p1, p2) × KE
const WWORDS = Math.ceil(P / 2); // weights as packed f16 pairs
const GS = WOFF + WWORDS;        // u32 words per genome
const GATHER_K = 64;             // genomes sampled per gene-bank pass
const PLANT_MAX = 600;

function clampInt(v, lo, hi, d) { const n = parseInt(v, 10); return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d; }
const $ = id => document.getElementById(id);
function fail(msg) { console.error(msg); const e = $('err'); e.textContent = msg; e.style.display = 'grid'; }

const common = (N) => /* wgsl */`
const F: u32 = ${F}u;
const FS: u32 = ${FS}u;
const KE: u32 = ${KE}u;
const NT: u32 = ${NT}u;
const HMAX: u32 = ${HMAX}u;
const G: u32 = ${G}u;
const GI: i32 = ${G};
const GF: f32 = ${G}.0;
const N: u32 = ${N}u;
const NI: u32 = ${NI}u;
const NO: u32 = ${NO}u;
const B1: u32 = ${NI * HMAX}u;
const W2: u32 = ${NI * HMAX + HMAX}u;
const B2: u32 = ${NI * HMAX + HMAX + HMAX * NO}u;
const HDR: u32 = ${HDR}u;
const EOFF: u32 = ${EOFF}u;
const WOFF: u32 = ${WOFF}u;
const GS: u32 = ${GS}u;
const WWORDS: u32 = ${WWORDS}u;
const MEM0: u32 = ${4 + F}u;
const GATE0: u32 = ${6 + F}u;
const DSCALE: f32 = 262144.0;
const INFO_GAIN: f32 = 10.0;

struct Particle { pos: vec2f, vel: vec2f, energy: f32, age: f32, alive: u32, kids: u32, mem: vec2f };

fn wrapD(d: vec2f) -> vec2f { return d - round(d / GF) * GF; }
fn wrapP(p: vec2f) -> vec2f { return p - floor(p / GF) * GF; }
fn cellIdx(x: i32, y: i32) -> u32 {
  let xx = ((x % GI) + GI) % GI;
  let yy = ((y % GI) + GI) % GI;
  return u32(yy) * G + u32(xx);
}
fn hsv(h: f32, s: f32, v: f32) -> vec3f {
  let k = vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0);
  let p = abs(fract(vec3f(h) + k) * 6.0 - 3.0);
  return v * mix(vec3f(1.0), clamp(p - 1.0, vec3f(0.0), vec3f(1.0)), s);
}
`;

const computeSrc = (N) => common(N) + /* wgsl */`
struct U {
  seedProb: f32, sun: f32, mouseOn: u32, metab: f32,
  mouse: vec2f, mouseR: f32, paramCost: f32,
  mutScale: f32, pressure: f32, plantCount: u32, plantR: f32,
  plantPos: vec2f, cohesion: f32, pad1: f32,
  src: array<vec4f, ${SOURCES}>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(2) var<storage, read_write> genome: array<u32>;
@group(0) @binding(3) var<storage, read_write> field: array<f32>;
@group(0) @binding(4) var<storage, read_write> fieldOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> deposit: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> reqs: array<vec2u>;      // split requests; also host upload area
@group(0) @binding(7) var<storage, read_write> freeList: array<u32>;   // free slots; also host download area
@group(0) @binding(8) var<storage, read_write> counters: array<atomic<u32>, 32>;
// counters: 0 split requests, 1 free slots, 2 alive, 3 births (total), 4 deaths (total),
//           5 sum active params, 6 sum hidden width, 7 max generation, 8 step counter, 9 lineages seeded,
//           16..16+NT effect-type histogram (slots in living particles)

var<private> rs: u32;
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd() -> f32 { rs = pcg(rs); return f32(rs >> 8u) / 16777216.0; }
fn rndn() -> f32 { return (rnd() + rnd() + rnd() - 1.5) * 2.0; }
fn sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
// genomes are interleaved (word k of slot s at k*N + s) so neighbouring threads read coalesced memory
fn gw(s: u32, k: u32) -> u32 { return genome[k * N + s]; }
fn sw(s: u32, k: u32, v: u32) { genome[k * N + s] = v; }
fn gf(s: u32, k: u32) -> f32 { return bitcast<f32>(genome[k * N + s]); }
fn sf(s: u32, k: u32, v: f32) { genome[k * N + s] = bitcast<u32>(v); }
fn wt(s: u32, k: u32) -> f32 {
  let w = unpack2x16float(genome[(WOFF + (k >> 1u)) * N + s]);
  return select(w.x, w.y, (k & 1u) == 1u);
}
fn uploadWord(w: u32) -> u32 { let v = reqs[w >> 1u]; return select(v.x, v.y, (w & 1u) == 1u); }
fn dep(cell: u32, c: u32, v: f32) { atomicAdd(&deposit[cell * FS + c], i32(round(v * DSCALE))); }
fn chanValue(c: u32) -> f32 { return select(1.0 / INFO_GAIN, 1.0, c == 0u); } // energy per unit in channel c
fn activeParams(h: u32) -> u32 { return NI * h + h + h * NO + NO; }
fn stepSeed(salt: u32) -> u32 { return pcg(atomicLoad(&counters[8]) * 2654435761u + salt); }
fn packEffect(t: u32, a: u32, b: u32) -> u32 { return t | (a << 8u) | (b << 16u); }

@compute @workgroup_size(1)
fn reset() {
  atomicStore(&counters[0], 0u);
  atomicStore(&counters[1], 0u);
  atomicStore(&counters[2], 0u);
  atomicStore(&counters[5], 0u);
  atomicStore(&counters[6], 0u);
  atomicStore(&counters[7], 0u);
  for (var t = 0u; t < NT; t++) { atomicStore(&counters[16u + t], 0u); }
  atomicAdd(&counters[8], 1u);
}

fn randomGenome(b: u32, lineage: u32) {
  sw(b, 0u, 2u + min(u32(rnd() * f32(HMAX - 1u)), HMAX - 2u));
  sf(b, 1u, 1.0 + rnd() * 6.0);              // sensing range (cells)
  sf(b, 2u, 0.01 + rnd() * 0.06);            // mutation rate
  sf(b, 3u, rnd());                          // lineage hue (drifts)
  sw(b, 4u, lineage);
  sw(b, 5u, 0u);                             // generation
  sf(b, 6u, 500.0 + rnd() * 2500.0);         // max age (steps)
  sf(b, 7u, 0.01 + rnd() * 0.2);             // internal clock frequency
  for (var c = 0u; c < F; c++) {
    sf(b, HDR + c, select(rnd() * 0.12, 0.02 + rnd() * 0.1, c == 0u));          // absorb rate
    sf(b, HDR + F + c, select(rndn() * 0.5, 0.5 + rnd() * 0.5, c == 0u));       // energy per unit absorbed
    sf(b, HDR + 2u * F + c, rndn() * 1.5);                                      // push along gradient
  }
  for (var s = 0u; s < KE; s++) {
    let t = select(0u, 1u + min(u32(rnd() * f32(NT - 1u)), NT - 2u), rnd() < 0.5);
    sw(b, EOFF + s * 4u, packEffect(t, min(u32(rnd() * f32(FS)), F), min(u32(rnd() * f32(F)), F - 1u)));
    sf(b, EOFF + s * 4u + 1u, rndn() * 0.6);
    sf(b, EOFF + s * 4u + 2u, rnd());
    sf(b, EOFF + s * 4u + 3u, rndn());
  }
  for (var k = 0u; k < WWORDS; k++) {
    sw(b, WOFF + k, pack2x16float(vec2f(rndn(), rndn()) * 0.45));
  }
}

fn newParticle(pos: vec2f, energy: f32) -> Particle {
  var p: Particle;
  p.pos = wrapP(pos);
  p.vel = vec2f(0.0);
  p.energy = energy;
  p.age = 0.0;
  p.alive = 1u;
  p.kids = 0u;
  p.mem = vec2f(0.0);
  return p;
}

@compute @workgroup_size(128)
fn seed(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= N) { return; }
  if (parts[i].alive == 1u) { return; }
  rs = stepSeed(i * 3u + 17u);
  if (rnd() >= u.seedProb) { return; }
  randomGenome(i, pcg(i + atomicAdd(&counters[9], 1u) * 977u));
  var p = newParticle(vec2f(rnd(), rnd()) * GF, 1.0);
  p.age = rnd() * 100.0;
  parts[i] = p;
}

fn die(i: u32, cell: u32, e: f32) {
  if (e > 0.0) { dep(cell, 0u, e); }    // remaining energy returns to the field
  parts[i].alive = 0u;
  atomicAdd(&counters[4], 1u);
}

@compute @workgroup_size(128)
fn step(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= N) { return; }
  var p = parts[i];
  if (p.alive == 0u) { return; }
  rs = stepSeed(i);
  let b = i;
  let x = i32(floor(p.pos.x));
  let y = i32(floor(p.pos.y));
  let here = cellIdx(x, y);

  if (u.mouseOn == 1u) {
    let d = wrapD(p.pos - u.mouse);
    if (dot(d, d) < u.mouseR * u.mouseR) { die(i, here, 0.0); return; }
  }

  // --- sense: field values + gradients at an evolved range (incl. density channel) ---
  let H = min(gw(b, 0u), HMAX);
  let r = i32(clamp(round(gf(b, 1u)), 1.0, 10.0));
  let cl = cellIdx(x - r, y); let cr = cellIdx(x + r, y);
  let cd = cellIdx(x, y - r); let cu = cellIdx(x, y + r);
  let inv = 1.0 / f32(2 * r);
  var inp: array<f32, ${NI}>;
  var val: array<f32, ${FS}>;
  var grd: array<vec2f, ${FS}>;
  for (var c = 0u; c < FS; c++) {
    let v = field[here * FS + c];
    let g = vec2f(field[cr * FS + c] - field[cl * FS + c], field[cu * FS + c] - field[cd * FS + c]) * inv;
    val[c] = v; grd[c] = g;
    let k = select(1.0, 0.3, c == F);
    inp[c] = v * k;
    inp[FS + c] = g.x * 8.0 * k;
    inp[2u * FS + c] = g.y * 8.0 * k;
  }
  inp[3u * FS] = p.energy - 1.0;
  inp[3u * FS + 1u] = p.vel.x * 2.0;
  inp[3u * FS + 2u] = p.vel.y * 2.0;
  inp[3u * FS + 3u] = p.mem.x;
  inp[3u * FS + 4u] = p.mem.y;
  inp[3u * FS + 5u] = sin(p.age * gf(b, 7u));

  // --- think: this particle's own net ---
  var hid: array<f32, ${HMAX}>;
  for (var h = 0u; h < H; h++) {
    var s = wt(b, B1 + h);
    for (var j = 0u; j < NI; j++) { s += wt(b, j * HMAX + h) * inp[j]; }
    hid[h] = tanh(s);
  }
  var o: array<f32, ${NO}>;
  for (var k = 0u; k < NO; k++) {
    var s = wt(b, B2 + k);
    for (var h = 0u; h < H; h++) { s += wt(b, W2 + h * NO + k) * hid[h]; }
    o[k] = s;
  }
  p.mem = vec2f(tanh(o[MEM0]), tanh(o[MEM0 + 1u]));

  var e = p.energy;

  // --- effect genes: mutable physics primitives, each gated by the net ---
  let gDen = grd[F];
  var force = vec2f(0.0);
  var anchor = 0.0;
  var heat = 0.0;
  var shield = 0.0;
  for (var s = 0u; s < KE; s++) {
    let base = EOFF + s * 4u;
    let tw = gw(b, base);
    let t = tw & 255u;
    if (t == 0u || t >= NT) { continue; }
    let ca = min((tw >> 8u) & 255u, F);          // may be the density channel
    let cb = min((tw >> 16u) & 255u, F - 1u);
    let cs = min(ca, F - 1u);                     // ca restricted to real channels
    let p0 = gf(b, base + 1u);
    let p1 = gf(b, base + 2u);
    let gate = sig(o[GATE0 + s]);
    // re-read from the (cache-hot) field rather than dynamically indexing private arrays, which spills them
    let vA = field[here * FS + ca];
    let vS = field[here * FS + cs];
    let ga = vec2f(field[cr * FS + ca] - field[cl * FS + ca], field[cu * FS + ca] - field[cd * FS + ca]) * inv;
    let na = ga / (length(ga) + 0.01);
    e -= 0.00008;                                 // upkeep per expressed effect
    switch t {
      case 1u: { force += p0 * gate * na; }                                   // taxis
      case 2u: { force += p0 * gate * vec2f(-na.y, na.x); }                   // swirl
      case 3u: { force += abs(p0) * gate * tanh((p1 * 2.0 - vA) * 6.0) * na; } // isoline seeking
      case 4u: {                                                              // catalyse a -> b (value-conserving)
        let amt = clamp(abs(p0), 0.0, 1.0) * gate * 0.1 * max(vS, 0.0);
        let vin = amt * chanValue(cs);
        let keep = clamp(p1, 0.0, 0.5);
        dep(here, cs, -amt);
        dep(here, cb, vin * (1.0 - keep) / chanValue(cb));
        e += vin * keep;
      }
      case 5u: {                                                              // jet
        let sp = length(p.vel);
        let hd = select(vec2f(cos(p1 * 6.2832), sin(p1 * 6.2832)), p.vel / max(sp, 1e-4), sp > 1e-3);
        force += abs(p0) * gate * hd * 0.6;
        e -= 0.002 * abs(p0) * gate;
      }
      case 6u: { anchor = max(anchor, clamp(abs(p0), 0.0, 1.0) * gate); }     // anchor / sessile
      case 7u: {                                                              // pressure (+) or cohesion (−)
        force -= p0 * gate * gDen / (length(gDen) + 0.2);
      }
      case 8u: {                                                              // pulsed emission into b
        let wave = 0.5 + 0.5 * sin(p.age * (0.02 + abs(p1) * 0.3));
        let amt = clamp(abs(p0), 0.0, 1.0) * gate * 0.02 * clamp(e, 0.0, 1.5) * wave;
        e -= amt;
        dep(here, cb, amt / chanValue(cb));
      }
      case 9u: {                                                              // siphon a at sensing range
        let off = na * f32(r);
        let fc = cellIdx(x + i32(round(off.x)), y + i32(round(off.y)));
        let amt = clamp(abs(p0), 0.0, 1.0) * gate * 0.15 * max(field[fc * FS + cs], 0.0);
        dep(fc, cs, -amt);
        e += amt * chanValue(cs) * clamp(p1, 0.0, 1.0);
      }
      case 10u: {                                                             // relay: copy signal a into b
        let amt = clamp(abs(p0), 0.0, 1.0) * gate * 0.2 * max(vS, 0.0);
        let cost = amt * chanValue(cb);
        if (cost < e * 0.05) { e -= cost; dep(here, cb, amt); }
      }
      case 11u: { heat += abs(p0) * gate * 0.3; }                             // heat (random walk)
      case 12u: {                                                             // toxin shield
        shield = max(shield, clamp(abs(p0), 0.0, 1.0) * gate);
        e -= 0.0005 * shield;
      }
      case 13u: { force += abs(p0) * gate * vec2f(cos(p1 * 6.2832), sin(p1 * 6.2832)) * 0.3; } // drift
      default: {}
    }
  }

  // --- chemistry: absorbing the field changes energy and pushes velocity ---
  var push = vec2f(0.0);
  for (var c = 0u; c < F; c++) {
    let a = gf(b, HDR + c) * max(val[c], 0.0);
    if (a > 0.0) {
      dep(here, c, -a);
      // info channels hold 1/INFO_GAIN energy per unit, so absorbing them can't create energy
      var de = gf(b, HDR + F + c) * a * chanValue(c);
      if (de < 0.0) { de *= 1.0 - shield; }
      e += de;
      let g = grd[c];
      push += gf(b, HDR + 2u * F + c) * tanh(a * 60.0) * g / (length(g) + 0.01);
    }
  }

  // --- write: spend energy to deposit along an F-dim aim ---
  if (o[2] > 0.0) {
    let amt = sig(o[3]) * 0.04 * clamp(e, 0.0, 1.5);
    var l1 = 0.0;
    for (var c = 0u; c < F; c++) { l1 += abs(o[4u + c]); }
    let s = amt / (l1 + 1e-4);
    for (var c = 0u; c < F; c++) { dep(here, c, o[4u + c] * s / chanValue(c)); }
    e -= amt;
  }

  // --- metabolism (bigger nets cost more); the spent energy leaves as exhaust in the info channels ---
  let burn = u.metab + u.paramCost * f32(activeParams(H));
  e -= burn;
  var l1i = 0.0;
  for (var c = 1u; c < F; c++) { l1i += abs(o[4u + c]); }
  let ex = burn * INFO_GAIN / (l1i + 1e-4);
  for (var c = 1u; c < F; c++) { dep(here, c, o[4u + c] * ex); }

  p.age += 1.0;
  if (e > 3.0) { dep(here, 0u, e - 3.0); e = 3.0; }
  if (e <= 0.0 || p.age > gf(b, 6u)) { die(i, here, e); return; }

  // --- split ---
  if (o[0] > 0.0 && e > 0.5 && p.age > 16.0) {
    let k = atomicAdd(&counters[0], 1u);
    if (k < N) { reqs[k] = vec2u(i, bitcast<u32>(0.15 + 0.7 * sig(o[1]))); }
  }

  // --- move: chemistry + effects + baseline incompressibility ---
  // surface tension: short-range pressure (range 1) against cohesion at the evolved sensing range -> droplets
  let g1 = vec2f(field[cellIdx(x + 1, y) * FS + F] - field[cellIdx(x - 1, y) * FS + F],
                 field[cellIdx(x, y + 1) * FS + F] - field[cellIdx(x, y - 1) * FS + F]) * 0.5;
  force -= u.pressure * g1 / (length(g1) * 0.25 + 1.0);
  force += u.cohesion * gDen / (length(gDen) + 0.05);
  var v = p.vel * 0.85 + push * 0.25 + force * 0.12 + vec2f(rnd() - 0.5, rnd() - 0.5) * (0.04 + heat);
  v *= 1.0 - anchor * 0.9;
  let sp = length(v);
  if (sp > 1.2) { v = v * (1.2 / sp); }
  p.vel = v;
  p.pos = wrapP(p.pos + v);
  p.energy = e;
  parts[i] = p;
  // occupancy for the density channel (at the new position)
  atomicAdd(&deposit[cellIdx(i32(floor(p.pos.x)), i32(floor(p.pos.y))) * FS + F], i32(DSCALE));
}

var<workgroup> wsum: array<atomic<u32>, ${4 + NT}>;
@compute @workgroup_size(128)
fn collect(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li < ${4 + NT}u) { atomicStore(&wsum[li], 0u); }
  workgroupBarrier();
  let i = gid.x;
  if (i < N) {
    if (parts[i].alive == 0u) {
      let k = atomicAdd(&counters[1], 1u);
      freeList[k] = i;
    } else {
      let h = gw(i, 0u);
      atomicAdd(&wsum[0], 1u);
      atomicAdd(&wsum[1], activeParams(h));
      atomicAdd(&wsum[2], h);
      atomicMax(&wsum[3], gw(i, 5u));
      for (var s = 0u; s < KE; s++) {
        let t = min(gw(i, EOFF + s * 4u) & 255u, NT - 1u);
        atomicAdd(&wsum[4u + t], 1u);
      }
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    atomicAdd(&counters[2], atomicLoad(&wsum[0]));
    atomicAdd(&counters[5], atomicLoad(&wsum[1]));
    atomicAdd(&counters[6], atomicLoad(&wsum[2]));
    atomicMax(&counters[7], atomicLoad(&wsum[3]));
  }
  if (li >= 4u && li < ${4 + NT}u) { atomicAdd(&counters[12u + li], atomicLoad(&wsum[li])); }
}

// Copies genome sb into db with mutation. Works in place (sb == db) since each word is read before written.
fn mutateCopy(sb: u32, db: u32) {
  let mr = clamp(gf(sb, 2u) * exp(rndn() * 0.2), 0.002, 0.3);
  let m = mr * u.mutScale;
  var h = gw(sb, 0u);
  if (rnd() < 0.05 * u.mutScale) {             // architecture mutation: grow or shrink the hidden layer
    if (rnd() < 0.5) { h = max(h, 3u) - 1u; } else { h = min(h + 1u, HMAX); }
  }
  sw(db, 0u, h);
  sf(db, 1u, clamp(gf(sb, 1u) + rndn() * 0.3 * u.mutScale, 1.0, 10.0));
  sf(db, 2u, mr);
  var hue = gf(sb, 3u) + rndn() * 0.015 * u.mutScale;
  var lineage = gw(sb, 4u);
  if (rnd() < 0.002 * u.mutScale) { hue = rnd(); lineage = pcg(lineage + 1u); } // occasional big jump
  sf(db, 3u, fract(hue + 1.0));
  sw(db, 4u, lineage);
  sw(db, 5u, gw(sb, 5u) + 1u);
  sf(db, 6u, clamp(gf(sb, 6u) * exp(rndn() * 0.05 * u.mutScale), 150.0, 8000.0));
  sf(db, 7u, clamp(gf(sb, 7u) * exp(rndn() * m), 0.002, 1.0));
  for (var c = 0u; c < F; c++) {
    let k = HDR + c;
    sf(db, k, clamp(gf(sb, k) + rndn() * m * 0.08, 0.0, 0.25));
    sf(db, k + F, clamp(gf(sb, k + F) + rndn() * m, -1.0, 1.0));
    sf(db, k + 2u * F, clamp(gf(sb, k + 2u * F) + rndn() * m * 2.0, -4.0, 4.0));
  }
  for (var s = 0u; s < KE; s++) {                // effect genes: retype, rewire, retune
    let base = EOFF + s * 4u;
    let tw = gw(sb, base);
    var t = tw & 255u; var a = (tw >> 8u) & 255u; var c2 = (tw >> 16u) & 255u;
    if (rnd() < 0.03 * u.mutScale) { t = min(u32(rnd() * f32(NT)), NT - 1u); }
    if (rnd() < 0.04 * u.mutScale) { a = min(u32(rnd() * f32(FS)), F); }
    if (rnd() < 0.04 * u.mutScale) { c2 = min(u32(rnd() * f32(F)), F - 1u); }
    sw(db, base, packEffect(t, a, c2));
    sf(db, base + 1u, clamp(gf(sb, base + 1u) + rndn() * m * 1.5, -1.5, 1.5));
    sf(db, base + 2u, clamp(gf(sb, base + 2u) + rndn() * m, -2.0, 2.0));
    sf(db, base + 3u, clamp(gf(sb, base + 3u) + rndn() * m, -2.0, 2.0));
  }
  for (var k = 0u; k < WWORDS; k++) {
    let w = unpack2x16float(gw(sb, WOFF + k)) + vec2f(rndn(), rndn()) * m;
    sw(db, WOFF + k, pack2x16float(clamp(w, vec2f(-8.0), vec2f(8.0))));
  }
}

@compute @workgroup_size(128)
fn spawn(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = min(min(atomicLoad(&counters[0]), atomicLoad(&counters[1])), N);
  if (i >= n) { return; }
  let rq = reqs[i];
  let pi = rq.x;
  let slot = freeList[i];
  rs = stepSeed(i * 7919u + 1234567u);
  var pp = parts[pi];
  let ce = pp.energy * bitcast<f32>(rq.y);
  pp.energy -= ce;
  pp.kids += 1u;
  parts[pi] = pp;
  var c = newParticle(pp.pos + vec2f(rnd() - 0.5, rnd() - 0.5) * 1.5, ce);
  c.vel = pp.vel * 0.5 + vec2f(rnd() - 0.5, rnd() - 0.5) * 0.2;
  parts[slot] = c;
  mutateCopy(pi, slot);
  atomicAdd(&counters[3], 1u);
}

// Host -> GPU: plant mutated copies of one uploaded genome into slots freed by the previous step.
@compute @workgroup_size(64)
fn plant(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  let used = min(min(atomicLoad(&counters[0]), atomicLoad(&counters[1])), N);
  let avail = atomicLoad(&counters[1]) - used;
  if (j >= min(u.plantCount, avail)) { return; }
  let slot = freeList[used + j];
  rs = stepSeed(j * 2971u + 99991u);
  for (var w = 0u; w < GS; w++) { sw(slot, w, uploadWord(w)); }
  mutateCopy(slot, slot);
  let ang = rnd() * 6.2832;
  let rad = sqrt(rnd()) * u.plantR;
  parts[slot] = newParticle(u.plantPos + vec2f(cos(ang), sin(ang)) * rad, 1.0);
}

// GPU -> host: copy sampled genomes (slot ids uploaded in reqs) into freeList for readback.
@compute @workgroup_size(128)
fn gather(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  let K = u.plantCount;
  if (t >= K * GS) { return; }
  let j = t / GS;
  let w = t % GS;
  let slot = min(uploadWord(j), N - 1u);
  freeList[t] = genome[w * N + slot];
  if (w == 0u) {
    let p = parts[slot];
    let o = K * GS + j * 4u;
    freeList[o] = p.alive;
    freeList[o + 1u] = p.kids;
    freeList[o + 2u] = bitcast<u32>(p.age);
    freeList[o + 3u] = bitcast<u32>(p.energy);
  }
}

@compute @workgroup_size(16, 16)
fn fieldStep(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= G || gid.y >= G) { return; }
  let x = i32(gid.x); let y = i32(gid.y);
  let id = cellIdx(x, y);
  let l = cellIdx(x - 1, y); let r = cellIdx(x + 1, y);
  let dn = cellIdx(x, y - 1); let up = cellIdx(x, y + 1);
  let wp = vec2f(f32(x) + 0.5, f32(y) + 0.5);
  var sun = 0.0;
  for (var s = 0u; s < ${SOURCES}u; s++) {
    let S = u.src[s];
    let d = wrapD(wp - S.xy);
    sun += S.w * exp(-dot(d, d) / (S.z * S.z));
  }
  for (var c = 0u; c < FS; c++) {
    let v = field[id * FS + c];
    let lap = field[l * FS + c] + field[r * FS + c] + field[dn * FS + c] + field[up * FS + c] - 4.0 * v;
    let d = f32(atomicExchange(&deposit[id * FS + c], 0)) / DSCALE;
    if (c == F) {                          // density: current occupancy, lightly smoothed
      fieldOut[id * FS + c] = clamp(d * 0.6 + (v + 0.2 * lap) * 0.4, 0.0, 64.0);
      continue;
    }
    let fc = f32(c) / f32(F);
    let diff = select(0.03 + 0.17 * fract(fc * 2.618 + 0.3), 0.04, c == 0u);
    let decay = select(0.004 + 0.03 * fract(fc * 1.618 + 0.1), 0.0006, c == 0u);
    var nv = v + diff * lap - decay * v + d;
    if (c == 0u) { nv += sun * u.sun; }
    fieldOut[id * FS + c] = clamp(nv, 0.0, 4.0);
  }
}
`;

const renderSrc = (N) => common(N) + /* wgsl */`
struct Cam { center: vec2f, scale: f32, mode: u32, screen: vec2f, pSize: f32, gain: f32 };
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> parts: array<Particle>;
@group(0) @binding(2) var<storage, read> genome: array<u32>;
@group(0) @binding(3) var<storage, read> field: array<f32>;

@vertex fn vsFull(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pts = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pts[vi], 0.0, 1.0);
}

fn pal(c: u32) -> vec3f {
  if (c == 0u) { return vec3f(0.55, 0.42, 0.08); }
  return hsv(fract(f32(c - 1u) / f32(F - 1u) + 0.45), 0.85, 1.0);
}
fn fval(cell: u32, c: u32) -> f32 { return field[cell * FS + c]; }

@fragment fn fsField(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  if (cam.mode == 1u) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let w = cam.center + (fc.xy - cam.screen * 0.5) / cam.scale - 0.5;
  let i0 = floor(w); let t = w - i0;
  let x = i32(i0.x); let y = i32(i0.y);
  let a = cellIdx(x, y); let b = cellIdx(x + 1, y); let c2 = cellIdx(x, y + 1); let d = cellIdx(x + 1, y + 1);
  if (cam.mode == 4u) {
    let v = mix(mix(fval(a, F), fval(b, F), t.x), mix(fval(c2, F), fval(d, F), t.x), t.y);
    return vec4f(vec3f(1.0 - exp(-v * 0.4)) * vec3f(0.8, 0.9, 1.0), 1.0);
  }
  var col = vec3f(0.0);
  for (var c = 0u; c < F; c++) {
    if (cam.mode == 3u && c != 0u) { continue; }
    let v = mix(mix(fval(a, c), fval(b, c), t.x), mix(fval(c2, c), fval(d, c), t.x), t.y);
    let k = select(cam.gain * 3.0, select(0.15, 0.6, cam.mode == 3u), c == 0u);
    col += pal(c) * v * k;
  }
  col = 1.0 - exp(-col);
  return vec4f(col, 1.0);
}

struct VO { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) uv: vec2f };
@vertex fn vsP(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO;
  let p = parts[ii];
  if (p.alive == 0u || cam.mode >= 2u) { o.pos = vec4f(4.0, 4.0, 0.0, 1.0); return o; }
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                                vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let cn = corners[vi];
  let d = wrapD(p.pos - cam.center);
  let s = d * cam.scale + cn * cam.pSize;
  o.pos = vec4f(s.x / (cam.screen.x * 0.5), -s.y / (cam.screen.y * 0.5), 0.0, 1.0);
  // colour = phenotype: chemistry genes + expressed effect types, projected to a hue; tinted by lineage
  var q = vec2f(0.0);
  for (var c = 0u; c < F; c++) {
    let a = f32(c) * 2.39996;
    let eg = bitcast<f32>(genome[(HDR + F + c) * N + ii]);
    let pg = bitcast<f32>(genome[(HDR + 2u * F + c) * N + ii]);
    q += vec2f(cos(a), sin(a)) * eg + vec2f(-sin(a), cos(a)) * pg * 0.35;
  }
  for (var k = 0u; k < KE; k++) {
    let t = genome[(EOFF + k * 4u) * N + ii] & 255u;
    if (t > 0u) {
      let p0 = bitcast<f32>(genome[(EOFF + k * 4u + 1u) * N + ii]);
      let a = f32(t) * 0.897;
      q += vec2f(cos(a), sin(a)) * min(abs(p0), 1.0) * 0.6;
    }
  }
  let lin = bitcast<f32>(genome[3u * N + ii]);
  let hue = fract(atan2(q.y, q.x) / 6.28318 + lin * 0.35);
  o.col = hsv(hue, clamp(0.35 + length(q) * 0.25, 0.0, 0.9), 0.25 + 0.55 * clamp(p.energy / 1.5, 0.0, 1.0));
  o.uv = cn;
  return o;
}
@fragment fn fsP(in: VO) -> @location(0) vec4f {
  let a = 1.0 - smoothstep(0.35, 1.0, length(in.uv));
  return vec4f(in.col * a, a);
}
`;

// ---------------------------------------------------------------------------------------------
// Gene bank: an archive of evolved genomes, indexed by a phenotype embedding (nearest-neighbour niches).
// ---------------------------------------------------------------------------------------------
const f16buf = new DataView(new ArrayBuffer(4));
function f16(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? 0 : s * 65504;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
function asF32(u) { f16buf.setUint32(0, u, true); return f16buf.getFloat32(0, true); }

// Fixed random ±1 projections of the weight vector: a cheap fingerprint of the net itself.
const PROJ_D = 8;
const projSigns = (() => {
  let s = 0x9e3779b9; const out = new Int8Array(PROJ_D * P);
  for (let i = 0; i < out.length; i++) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; out[i] = (s & 1) ? 1 : -1; }
  return out;
})();

function decodeGenome(g) {
  const H = Math.min(g[0], HMAX);
  const chem = { absorb: [], egain: [], push: [] };
  for (let c = 0; c < F; c++) {
    chem.absorb.push(asF32(g[HDR + c])); chem.egain.push(asF32(g[HDR + F + c])); chem.push.push(asF32(g[HDR + 2 * F + c]));
  }
  const effects = [];
  for (let s = 0; s < KE; s++) {
    const tw = g[EOFF + s * 4];
    effects.push({ t: Math.min(tw & 255, NT - 1), a: (tw >> 8) & 255, b: (tw >> 16) & 255,
      p0: asF32(g[EOFF + s * 4 + 1]), p1: asF32(g[EOFF + s * 4 + 2]) });
  }
  return { H, params: NI * H + H + H * NO + NO, range: asF32(g[1]), mut: asF32(g[2]), hue: asF32(g[3]),
    lineage: g[4] >>> 0, gen: g[5], maxAge: asF32(g[6]), clock: asF32(g[7]), chem, effects };
}

function embed(g, d) {
  const v = [];
  for (let c = 0; c < F; c++) v.push(d.chem.egain[c], d.chem.push[c] * 0.25, d.chem.absorb[c] * 4);
  const eff = new Array(NT).fill(0);
  for (const e of d.effects) if (e.t > 0) eff[e.t] += Math.min(Math.abs(e.p0), 1.5);
  v.push(...eff.slice(1).map(x => x * 0.8));
  v.push(d.range / 10, d.H / HMAX, Math.min(d.clock, 1));
  const w = new Float32Array(P);
  for (let k = 0; k < WWORDS; k++) { const u = g[WOFF + k]; w[2 * k] = f16(u & 0xffff); if (2 * k + 1 < P) w[2 * k + 1] = f16(u >>> 16); }
  const norm = 1 / Math.sqrt(P);
  for (let j = 0; j < PROJ_D; j++) {
    let s = 0; const o = j * P;
    for (let k = 0; k < P; k++) s += w[k] * projSigns[o + k];
    v.push(Math.tanh(s * norm * 0.5));
  }
  return Float32Array.from(v);
}

function phenoColor(d) {
  let qx = 0, qy = 0;
  for (let c = 0; c < F; c++) {
    const a = c * 2.39996;
    qx += Math.cos(a) * d.chem.egain[c] - Math.sin(a) * d.chem.push[c] * 0.35;
    qy += Math.sin(a) * d.chem.egain[c] + Math.cos(a) * d.chem.push[c] * 0.35;
  }
  for (const e of d.effects) if (e.t > 0) { const a = e.t * 0.897, m = Math.min(Math.abs(e.p0), 1) * 0.6; qx += Math.cos(a) * m; qy += Math.sin(a) * m; }
  const hue = (((Math.atan2(qy, qx) / 6.28318 + d.hue * 0.35) % 1) + 1) % 1;
  return `hsl(${(hue * 360).toFixed(0)} 80% 62%)`;
}

class GeneBank {
  constructor(cap = 400) { this.cap = cap; this.entries = []; this.radius = 0.6; this.inserts = 0; this.replaced = 0; this.nextId = 1; }
  dist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
  nearest(v) {
    let best = -1, bd = Infinity;
    this.entries.forEach((e, i) => { const d = this.dist(v, e.vec); if (d < bd) { bd = d; best = i; } });
    return [best, bd];
  }
  add(genome, meta) {
    const dec = decodeGenome(genome);
    const vec = embed(genome, dec);
    const fitness = meta.kids + meta.age / 2000;
    const [i, d] = this.nearest(vec);
    this.inserts++;
    if (i >= 0 && d < this.radius) {
      if (fitness > this.entries[i].fitness) {
        Object.assign(this.entries[i], { genome, dec, vec, fitness, kids: meta.kids, color: phenoColor(dec), t: Date.now() });
        this.replaced++;
      }
      return;
    }
    this.entries.push({ id: this.nextId++, genome, dec, vec, fitness, kids: meta.kids, color: phenoColor(dec), t: Date.now() });
    if (this.entries.length > this.cap) this.prune();
  }
  prune() { // drop the weaker member of the closest pair, and widen the niche radius
    let bi = 0, bj = 1, bd = Infinity;
    const E = this.entries;
    for (let i = 0; i < E.length; i++) for (let j = i + 1; j < E.length; j++) {
      const d = this.dist(E[i].vec, E[j].vec); if (d < bd) { bd = d; bi = i; bj = j; }
    }
    E.splice(E[bi].fitness < E[bj].fitness ? bi : bj, 1);
    this.radius = Math.max(this.radius, bd * 1.02);
  }
  pca() { // top-2 principal components by power iteration
    const E = this.entries; if (E.length < 3) return null;
    const D = E[0].vec.length, mean = new Float64Array(D);
    for (const e of E) for (let i = 0; i < D; i++) mean[i] += e.vec[i] / E.length;
    const X = E.map(e => Array.from(e.vec, (x, i) => x - mean[i]));
    const comps = [];
    for (let k = 0; k < 2; k++) {
      let v = Array.from({ length: D }, (_, i) => Math.sin(i * 7.1 + k * 3.3));
      for (let it = 0; it < 30; it++) {
        const nv = new Array(D).fill(0);
        for (const x of X) { let s = 0; for (let i = 0; i < D; i++) s += x[i] * v[i]; for (let i = 0; i < D; i++) nv[i] += s * x[i]; }
        for (const c of comps) { let s = 0; for (let i = 0; i < D; i++) s += nv[i] * c[i]; for (let i = 0; i < D; i++) nv[i] -= s * c[i]; }
        const n = Math.hypot(...nv) || 1; v = nv.map(x => x / n);
      }
      comps.push(v);
    }
    return X.map(x => comps.map(c => x.reduce((s, xi, i) => s + xi * c[i], 0)));
  }
  serialize() {
    return JSON.stringify({ v: 2, F, HMAX, GS, radius: this.radius, entries: this.entries.map(e => ({
      id: e.id, fitness: e.fitness, kids: e.kids, g: btoa(String.fromCharCode(...new Uint8Array(e.genome.buffer))) })) });
  }
  load(json) {
    const o = typeof json === 'string' ? JSON.parse(json) : json;
    if (o.F !== F || o.HMAX !== HMAX || o.GS !== GS) throw new Error(`Gene bank was saved with F=${o.F}, HMAX=${o.HMAX}; this run uses F=${F}, HMAX=${HMAX}.`);
    this.entries = o.entries.map(e => {
      const bytes = Uint8Array.from(atob(e.g), ch => ch.charCodeAt(0));
      const genome = new Uint32Array(bytes.buffer);
      const dec = decodeGenome(genome);
      return { id: e.id, genome, dec, vec: embed(genome, dec), fitness: e.fitness, kids: e.kids, color: phenoColor(dec), t: Date.now() };
    });
    this.radius = o.radius || 0.6;
    this.nextId = Math.max(1, ...this.entries.map(e => e.id + 1));
  }
}

async function main() {
  if (!navigator.gpu) return fail('WebGPU is not available in this browser.\nUse a recent Chrome or Edge with hardware acceleration enabled.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No WebGPU adapter found.');
  const lim = adapter.limits;
  const maxBind = Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize);
  let N = Math.min(N_REQ, Math.floor(maxBind / (GS * 4)));
  N = Math.floor(N / 128) * 128;
  if (G * G * FS * 4 > maxBind) return fail(`Field ${G}×${G}×${FS} exceeds this GPU's buffer limit.`);
  const ainfo = adapter.info || {};
  console.log('adapter', ainfo.vendor, ainfo.architecture, ainfo.description, 'N', N, 'GS', GS, 'P', P);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: maxBind,
      maxBufferSize: lim.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 10),
    },
  });
  device.addEventListener('uncapturederror', ev => fail('GPU error: ' + ev.error.message));
  device.lost.then(info => fail('GPU device lost: ' + info.message));

  const canvas = $('c');
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const cmod = device.createShaderModule({ code: computeSrc(N) });
  const rmod = device.createShaderModule({ code: renderSrc(N) });
  for (const [name, m] of [['compute', cmod], ['render', rmod]]) {
    const info = await m.getCompilationInfo();
    const errs = info.messages.filter(x => x.type === 'error');
    if (errs.length) return fail(`${name} shader:\n` + errs.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
  }

  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const buf = (size, usage) => device.createBuffer({ size, usage });
  const UBYTES = 64 + SOURCES * 16;
  const uni = buf(UBYTES, GPUBufferUsage.UNIFORM | CD);
  const camBuf = buf(32, GPUBufferUsage.UNIFORM | CD);
  const parts = buf(N * 40, S | CD);
  const genome = buf(N * GS * 4, S | CD);
  const fieldBufs = [buf(G * G * FS * 4, S | CD | CS), buf(G * G * FS * 4, S | CD | CS)];
  const deposit = buf(G * G * FS * 4, S | CD);
  const reqs = buf(Math.max(N * 8, GS * 4 + 16), S | CD);
  const freeList = buf(Math.max(N * 4, (GATHER_K * (GS + 4)) * 4), S | CS);
  const counters = buf(128, S | CS | CD);
  const staging = buf(128, GPUBufferUsage.MAP_READ | CD);
  const GATHER_BYTES = GATHER_K * (GS + 4) * 4;
  const gatherRead = buf(GATHER_BYTES, GPUBufferUsage.MAP_READ | CD);

  const ST = { type: 'storage' }, RO = { type: 'read-only-storage' };
  const cLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ...[1, 2, 3, 4, 5, 6, 7, 8].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: ST }))],
  });
  const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
  const rLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      ...[1, 2, 3].map(binding => ({ binding, visibility: VF, buffer: RO }))],
  });
  const cpl = device.createPipelineLayout({ bindGroupLayouts: [cLayout] });
  const rpl = device.createPipelineLayout({ bindGroupLayouts: [rLayout] });
  const kernel = entryPoint => device.createComputePipeline({ layout: cpl, compute: { module: cmod, entryPoint } });
  const K = Object.fromEntries(['reset', 'seed', 'step', 'collect', 'spawn', 'fieldStep', 'plant', 'gather'].map(k => [k, kernel(k)]));

  const cGroups = [0, 1].map(cur => device.createBindGroup({
    layout: cLayout,
    entries: [uni, parts, genome, fieldBufs[cur], fieldBufs[1 - cur], deposit, reqs, freeList, counters]
      .map((buffer, binding) => ({ binding, resource: { buffer } })),
  }));
  const rGroups = [0, 1].map(cur => device.createBindGroup({
    layout: rLayout,
    entries: [camBuf, parts, genome, fieldBufs[cur]].map((buffer, binding) => ({ binding, resource: { buffer } })),
  }));

  const fieldPipe = device.createRenderPipeline({
    layout: rpl,
    vertex: { module: rmod, entryPoint: 'vsFull' },
    fragment: { module: rmod, entryPoint: 'fsField', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const add = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
  const partPipe = device.createRenderPipeline({
    layout: rpl,
    vertex: { module: rmod, entryPoint: 'vsP' },
    fragment: { module: rmod, entryPoint: 'fsP', targets: [{ format, blend: { color: add, alpha: add } }] },
    primitive: { topology: 'triangle-list' },
  });

  // ---------- host state ----------
  const ui = {};
  for (const id of ['sun', 'metab', 'pcost', 'mut', 'pressure', 'cohesion', 'spf', 'brush', 'fgain']) {
    const el = $(id), out = el.nextElementSibling;
    if (qs.get(id)) el.value = qs.get(id);
    const sync = () => { ui[id] = parseFloat(el.value); out.textContent = fmtSlider(id, ui[id]); };
    el.addEventListener('input', sync); sync();
  }
  function fmtSlider(id, v) {
    if (id === 'metab') return (v * 1000).toFixed(1) + 'e-3';
    if (id === 'pcost') return (v * 1e6).toFixed(1) + 'e-6';
    return String(v);
  }
  let mode = clampInt(qs.get('mode'), 0, 4, 0), paused = false, cur = 0, seedProb = 0.35, frameNo = 0;
  let tool = 'erase';
  const cam = { cx: G / 2, cy: G / 2, scale: 1 };
  const mouse = { x: 0, y: 0, down: false, panning: false, lx: 0, ly: 0 };
  const stats = { alive: 0, births: 0, deaths: 0, params: 0, hidden: 0, gen: 0, effects: new Array(NT).fill(0), pending: false };
  let stepsDone = 0, stepRate = 0, fps = 0, rateT = performance.now(), rateSteps = 0, rateFrames = 0;
  const bank = new GeneBank();
  const BANK_KEY = `fieldlife-bank-v2-F${F}-H${HMAX}`;
  try { const saved = localStorage.getItem(BANK_KEY); if (saved && !qs.has('fresh')) bank.load(saved); } catch (e) { console.warn('gene bank not restored', e); }
  let selected = null;
  const plantQ = [];            // { genome, x, y, r, count, stage }
  let gatherPending = false, lastGather = 0, lastImmig = performance.now(), lastSave = performance.now();

  const sources = Array.from({ length: SOURCES }, () => newSource());
  function newSource() {
    return { x: Math.random() * G, y: Math.random() * G, vx: 0, vy: 0, sigma: G * (0.07 + Math.random() * 0.1),
      base: 0.001 + Math.random() * 0.0015, phase: Math.random() * 6.28, freq: 0.0005 + Math.random() * 0.002 };
  }

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.floor(innerWidth * dpr);
    canvas.height = Math.floor(innerHeight * dpr);
    cam.fit = Math.max(canvas.width, canvas.height) / G;
    cam.scale = Math.max(cam.scale, cam.fit);
  }
  addEventListener('resize', resize);
  resize();
  cam.scale = cam.fit;

  const toWorld = (px, py) => {
    const dpr = canvas.width / innerWidth;
    return [cam.cx + (px * dpr - canvas.width / 2) / cam.scale, cam.cy + (py * dpr - canvas.height / 2) / cam.scale];
  };
  const brushWorld = () => ui.brush * (canvas.width / innerWidth) / cam.scale;
  function queuePlant(entry, x, y, r, count) {
    if (!entry || plantQ.length > 8) return;
    plantQ.push({ genome: entry.genome, x: ((x % G) + G) % G, y: ((y % G) + G) % G, r, count: Math.min(count, PLANT_MAX), stage: 1 });
  }
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId);
    mouse.lx = e.clientX; mouse.ly = e.clientY;
    [mouse.x, mouse.y] = toWorld(e.clientX, e.clientY);
    if (e.button !== 0) { mouse.panning = true; return; }
    if (tool === 'plant') {
      const sp = selected || bank.entries[Math.floor(Math.random() * bank.entries.length)];
      if (sp) { const r = brushWorld(); queuePlant(sp, mouse.x, mouse.y, r, Math.round(Math.PI * r * r * 0.6)); }
    } else mouse.down = true;
  });
  canvas.addEventListener('pointermove', e => {
    if (mouse.panning) {
      const dpr = canvas.width / innerWidth;
      cam.cx -= (e.clientX - mouse.lx) * dpr / cam.scale;
      cam.cy -= (e.clientY - mouse.ly) * dpr / cam.scale;
    }
    mouse.lx = e.clientX; mouse.ly = e.clientY;
    [mouse.x, mouse.y] = toWorld(e.clientX, e.clientY);
  });
  const up = () => { mouse.down = false; mouse.panning = false; };
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const [wx, wy] = toWorld(e.clientX, e.clientY);
    cam.scale = Math.min(cam.fit * 40, Math.max(cam.fit, cam.scale * Math.exp(-e.deltaY * 0.0015)));
    const [nx, ny] = toWorld(e.clientX, e.clientY);
    cam.cx += wx - nx; cam.cy += wy - ny;
  }, { passive: false });

  if (qs.has('noimm')) $('immigrate').checked = false;
  const modeSel = $('mode');
  modeSel.value = String(mode);
  modeSel.addEventListener('change', () => { mode = +modeSel.value; });
  const pauseBtn = $('pause');
  const togglePause = () => { paused = !paused; pauseBtn.textContent = paused ? 'play' : 'pause'; };
  pauseBtn.addEventListener('click', togglePause);
  $('seed').addEventListener('click', () => { seedProb = 0.1; });
  $('reset').addEventListener('click', resetWorld);
  $('collapse').addEventListener('click', () => {
    const p = $('panel'); p.classList.toggle('collapsed');
    $('collapse').textContent = p.classList.contains('collapsed') ? '+' : '–';
  });
  const setTool = t => { tool = t; $('toolErase').classList.toggle('on', t === 'erase'); $('toolPlant').classList.toggle('on', t === 'plant'); };
  $('toolErase').addEventListener('click', () => setTool('erase'));
  $('toolPlant').addEventListener('click', () => setTool('plant'));
  $('plantRandom').addEventListener('click', () => plantRandom());
  $('bankSave').addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([bank.serialize()], { type: 'application/json' }));
    a.download = `field-life-genebank-${bank.entries.length}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $('bankLoad').addEventListener('click', () => $('bankFile').click());
  $('bankFile').addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    try { bank.load(await f.text()); selected = null; drawBank(); } catch (err) { alert(err.message); }
    e.target.value = '';
  });
  $('bankClear').addEventListener('click', () => { bank.entries = []; bank.radius = 0.6; selected = null; drawBank(); saveBank(); });
  addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); togglePause(); }
    if (e.key >= '1' && e.key <= '5') { mode = +e.key - 1; modeSel.value = String(mode); }
    if (e.key === 'r') resetWorld();
    if (e.key === 'e') setTool('erase');
    if (e.key === 'p') setTool('plant');
  });
  function saveBank() { try { localStorage.setItem(BANK_KEY, bank.serialize()); } catch (e) { console.warn('gene bank not saved', e); } }
  addEventListener('beforeunload', saveBank);

  function plantRandom() {
    if (!bank.entries.length) return;
    // uniform over niches (not individuals), so rare strategies get re-introduced as often as common ones
    const sp = bank.entries[Math.floor(Math.random() * bank.entries.length)];
    queuePlant(sp, Math.random() * G, Math.random() * G, 14, 360);
  }

  function resetWorld() {
    const enc = device.createCommandEncoder();
    enc.clearBuffer(parts); enc.clearBuffer(fieldBufs[0]); enc.clearBuffer(fieldBufs[1]);
    enc.clearBuffer(deposit); enc.clearBuffer(counters);
    device.queue.submit([enc.finish()]);
    seedProb = 0.35;
    plantQ.length = 0;
  }

  // ----- gene bank map -----
  const map = $('map'), mctx = map.getContext('2d');
  let mapPts = [];
  function drawBank() {
    const W = map.width, H = map.height;
    mctx.fillStyle = '#050607'; mctx.fillRect(0, 0, W, H);
    const proj = bank.pca();
    mapPts = [];
    if (proj) {
      let mx = Infinity, Mx = -Infinity, my = Infinity, My = -Infinity;
      for (const [a, b] of proj) { mx = Math.min(mx, a); Mx = Math.max(Mx, a); my = Math.min(my, b); My = Math.max(My, b); }
      const sx = (W - 24) / ((Mx - mx) || 1), sy = (H - 24) / ((My - my) || 1);
      const maxFit = Math.max(1, ...bank.entries.map(e => e.fitness));
      bank.entries.forEach((e, i) => {
        const x = 12 + (proj[i][0] - mx) * sx, y = 12 + (proj[i][1] - my) * sy;
        mapPts.push([x, y, e]);
        const r = 2 + 5 * Math.sqrt(e.fitness / maxFit);
        mctx.fillStyle = e.color; mctx.globalAlpha = 0.85;
        mctx.beginPath(); mctx.arc(x, y, r, 0, 6.2832); mctx.fill();
        if (selected && e.id === selected.id) { mctx.globalAlpha = 1; mctx.strokeStyle = '#fff'; mctx.lineWidth = 2; mctx.beginPath(); mctx.arc(x, y, r + 4, 0, 6.2832); mctx.stroke(); }
      });
      mctx.globalAlpha = 1;
    } else {
      mctx.fillStyle = '#7a8a80'; mctx.font = '22px ui-monospace, monospace';
      mctx.fillText('collecting evolved nets…', 20, H / 2);
    }
    const gens = bank.entries.map(e => e.dec.gen);
    $('bankStats').textContent =
      `niches     ${bank.entries.length} / ${bank.cap}   radius ${bank.radius.toFixed(2)}\n` +
      `samples    ${bank.inserts.toLocaleString()}   upgrades ${bank.replaced}\n` +
      `generation ${gens.length ? Math.min(...gens) + '–' + Math.max(...gens) : '—'}`;
    showSpecies();
  }
  function showSpecies() {
    if (!selected) { $('species').textContent = 'click a dot to select a species'; return; }
    const d = selected.dec;
    const ch = c => (c >= F ? 'ρ' : c === 0 ? 'E' : 'c' + c);
    const effs = d.effects.filter(e => e.t > 0).map(e => {
      const n = EFFECTS[e.t];
      const two = ['catalyse', 'relay'].includes(n) ? `${ch(Math.min(e.a, F - 1))}→${ch(e.b)}` : ['pulse'].includes(n) ? `→${ch(e.b)}` : ['jet', 'anchor', 'heat', 'shield', 'drift'].includes(n) ? '' : ch(e.a);
      return `${n}${two ? '(' + two + ')' : ''} ${e.p0 >= 0 ? '+' : ''}${e.p0.toFixed(2)}`;
    });
    const eats = d.chem.egain.map((g, c) => [g * d.chem.absorb[c], c]).sort((a, b) => b[0] - a[0]);
    $('species').textContent =
      `#${selected.id}  gen ${d.gen}  kids ${selected.kids}  fit ${selected.fitness.toFixed(1)}\n` +
      `net ${NI}→${d.H}→${NO} (${d.params} params)  range ${d.range.toFixed(1)}\n` +
      `effects: ${effs.join(', ') || 'none'}\n` +
      `feeds on ${eats.slice(0, 2).map(([, c]) => ch(c)).join(', ')}; poisoned by ${eats.slice(-1).map(([, c]) => ch(c)).join('')}`;
  }
  map.addEventListener('click', e => {
    const r = map.getBoundingClientRect();
    const x = (e.clientX - r.left) * map.width / r.width, y = (e.clientY - r.top) * map.height / r.height;
    let best = null, bd = 400;
    for (const [px, py, en] of mapPts) { const d = (px - x) ** 2 + (py - y) ** 2; if (d < bd) { bd = d; best = en; } }
    selected = best;
    if (best) setTool('plant');
    drawBank();
  });
  drawBank();

  const uData = new ArrayBuffer(UBYTES);
  const uF = new Float32Array(uData), uU = new Uint32Array(uData);
  const camData = new ArrayBuffer(32);
  const camF = new Float32Array(camData), camU = new Uint32Array(camData);
  const wg = Math.ceil(N / 128), fg = Math.ceil(G / 16);

  function writeUniforms(extra = {}) {
    uF[0] = seedProb; uF[1] = ui.sun; uU[2] = extra.mouseOn ?? 0; uF[3] = ui.metab;
    uF[4] = extra.mx ?? 0; uF[5] = extra.my ?? 0; uF[6] = extra.mr ?? 0; uF[7] = ui.pcost;
    uF[8] = ui.mut; uF[9] = ui.pressure; uU[10] = extra.count ?? 0; uF[11] = extra.pr ?? 0;
    uF[12] = extra.px ?? 0; uF[13] = extra.py ?? 0; uF[14] = ui.cohesion;
    sources.forEach((s, k) => {
      uF[16 + k * 4] = s.x; uF[17 + k * 4] = s.y; uF[18 + k * 4] = s.sigma;
      uF[19 + k * 4] = s.base * (0.35 + 0.65 * (0.5 + 0.5 * Math.sin(s.phase)));
    });
    device.queue.writeBuffer(uni, 0, uData);
  }

  function runPrePass(now) {
    if (paused) return;
    // plant stage 2: the previous frame's steps freed the area; fill those slots with the uploaded genome
    const p2 = plantQ.find(p => p.stage === 2);
    if (p2) {
      plantQ.splice(plantQ.indexOf(p2), 1);
      device.queue.writeBuffer(reqs, 0, p2.genome);
      writeUniforms({ count: p2.count, px: p2.x, py: p2.y, pr: p2.r });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, cGroups[cur]); pass.setPipeline(K.plant); pass.dispatchWorkgroups(Math.ceil(p2.count / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
      return;
    }
    // gene bank sampling: gather random slots' genomes for readback
    if (!gatherPending && now - lastGather > 1000 && stats.alive > 0) {
      lastGather = now;
      gatherPending = true;
      const ids = new Uint32Array(GATHER_K);
      for (let j = 0; j < GATHER_K; j++) ids[j] = Math.floor(Math.random() * N);
      device.queue.writeBuffer(reqs, 0, ids);
      writeUniforms({ count: GATHER_K });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, cGroups[cur]); pass.setPipeline(K.gather); pass.dispatchWorkgroups(Math.ceil(GATHER_K * GS / 128));
      pass.end();
      enc.copyBufferToBuffer(freeList, 0, gatherRead, 0, GATHER_BYTES);
      device.queue.submit([enc.finish()]);
      gatherRead.mapAsync(GPUMapMode.READ).then(() => {
        const all = new Uint32Array(gatherRead.getMappedRange().slice(0));
        gatherRead.unmap();
        const metaOff = GATHER_K * GS;
        for (let j = 0; j < GATHER_K; j++) {
          const alive = all[metaOff + j * 4], kids = all[metaOff + j * 4 + 1];
          if (!alive || kids < 1) continue;     // only proven reproducers enter the bank
          const age = asF32(all[metaOff + j * 4 + 2]);
          bank.add(all.slice(j * GS, (j + 1) * GS), { kids, age });
        }
        if (selected && !bank.entries.includes(selected)) selected = bank.entries.find(e => e.id === selected.id) || null;
        gatherPending = false;
        drawBank();
      }).catch(() => { gatherPending = false; });
    }
  }

  function frame(now) {
    rateFrames++;
    const steps = paused ? 0 : ui.spf;
    if (!paused) {
      for (const s of sources) {
        s.vx = s.vx * 0.995 + (Math.random() - 0.5) * 0.02;
        s.vy = s.vy * 0.995 + (Math.random() - 0.5) * 0.02;
        s.x = (s.x + s.vx * steps + G) % G; s.y = (s.y + s.vy * steps + G) % G;
        s.phase += s.freq * steps;
      }
      if (stepsDone > 300 && stats.alive < N * 0.005 && frameNo % 30 === 0) {
        if (bank.entries.length) for (let k = 0; k < 4; k++) plantRandom();
        seedProb = Math.max(seedProb, 0.05);
      }
      if ($('immigrate').checked && bank.entries.length >= 8 && now - lastImmig > 15000) { lastImmig = now; plantRandom(); }
      if (now - lastSave > 30000) { lastSave = now; saveBank(); }
    }

    runPrePass(now);

    // main uniforms: plant stage 1 clears its area via the eraser path this frame
    const p1 = !paused && plantQ.find(p => p.stage === 1);
    if (p1) p1.stage = 2;
    if (p1) writeUniforms({ mouseOn: 1, mx: p1.x, my: p1.y, mr: p1.r });
    else writeUniforms({ mouseOn: mouse.down && !paused ? 1 : 0, mx: ((mouse.x % G) + G) % G, my: ((mouse.y % G) + G) % G, mr: brushWorld() });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    if (seedProb > 0 && !paused) {
      pass.setBindGroup(0, cGroups[cur]);
      pass.setPipeline(K.reset); pass.dispatchWorkgroups(1);
      pass.setPipeline(K.seed); pass.dispatchWorkgroups(wg);
      seedProb = 0;
    }
    for (let s = 0; s < steps; s++) {
      pass.setBindGroup(0, cGroups[cur]);
      pass.setPipeline(K.reset); pass.dispatchWorkgroups(1);
      pass.setPipeline(K.step); pass.dispatchWorkgroups(wg);
      pass.setPipeline(K.collect); pass.dispatchWorkgroups(wg);
      pass.setPipeline(K.spawn); pass.dispatchWorkgroups(wg);
      pass.setPipeline(K.fieldStep); pass.dispatchWorkgroups(fg, fg);
      cur = 1 - cur;
    }
    pass.end();
    stepsDone += steps; rateSteps += steps;

    cam.cx = ((cam.cx % G) + G) % G; cam.cy = ((cam.cy % G) + G) % G;
    camF[0] = cam.cx; camF[1] = cam.cy; camF[2] = cam.scale; camU[3] = mode;
    camF[4] = canvas.width; camF[5] = canvas.height; camF[6] = Math.max(0.9, cam.scale * 0.38); camF[7] = ui.fgain;
    device.queue.writeBuffer(camBuf, 0, camData);

    const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    rp.setBindGroup(0, rGroups[cur]);
    rp.setPipeline(fieldPipe); rp.draw(3);
    rp.setPipeline(partPipe); rp.draw(6, N);
    rp.end();

    const readStats = !stats.pending && frameNo % 6 === 0;
    if (readStats) enc.copyBufferToBuffer(counters, 0, staging, 0, 128);
    device.queue.submit([enc.finish()]);
    if (readStats) {
      stats.pending = true;
      staging.mapAsync(GPUMapMode.READ).then(() => {
        const c = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        stats.alive = c[2]; stats.births = c[3]; stats.deaths = c[4];
        stats.params = c[2] ? c[5] / c[2] : 0; stats.hidden = c[2] ? c[6] / c[2] : 0; stats.gen = c[7];
        stats.effects = Array.from(c.slice(16, 16 + NT));
        stats.pending = false;
      });
    }

    if (now - rateT > 500) {
      stepRate = rateSteps * 1000 / (now - rateT); fps = rateFrames * 1000 / (now - rateT);
      rateT = now; rateSteps = 0; rateFrames = 0;
      $('stats').textContent =
        `slots      ${N.toLocaleString()}\n` +
        `field      ${G}×${G}×${F} (+density)\n` +
        `net        ${NI} in → ≤${HMAX} hidden → ${NO} out\n` +
        `avg params ${stats.params.toFixed(0)}  (max ${P})\n` +
        `avg hidden ${stats.hidden.toFixed(2)}\n` +
        `max gen    ${stats.gen}\n` +
        `births     ${stats.births.toLocaleString()}\n` +
        `deaths     ${stats.deaths.toLocaleString()}\n` +
        `weights    ${(stats.alive * stats.params / 1e6).toFixed(1)}M live`;
      const tot = Math.max(1, stats.alive * KE);
      $('effects').textContent = EFFECTS.map((n, t) => [n, stats.effects[t] / tot]).filter(([n]) => n !== 'none')
        .sort((a, b) => b[1] - a[1])
        .map(([n, f]) => `${n.padEnd(9)} ${'█'.repeat(Math.round(f * 60)).padEnd(14)} ${(f * 100).toFixed(1)}%`).join('\n');
    }
    $('hud').textContent = `zoom=${(cam.scale / cam.fit).toFixed(1)}x  pop=${stats.alive}  ${stepRate.toFixed(1)} steps/s  ${fps.toFixed(0)} fps  ${canvas.width}px  bank=${bank.entries.length}${paused ? '  [paused]' : ''}`;
    frameNo++;
    requestAnimationFrame(frame);
  }

  // Debug helpers (used by headless checks).
  async function fieldSummary() {
    const size = G * G * FS * 4;
    const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(fieldBufs[cur], 0, rb, 0, size);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(rb.getMappedRange());
    const mean = new Array(FS).fill(0), max = new Array(FS).fill(0);
    for (let i = 0; i < f.length; i++) { const c = i % FS; mean[c] += f[i]; if (f[i] > max[c]) max[c] = f[i]; }
    rb.unmap(); rb.destroy();
    return { mean: mean.map(m => +(m / (G * G)).toFixed(4)), max: max.map(m => +m.toFixed(3)) };
  }
  window.__fieldLife = {
    N, G, F, P, GS, fieldSummary, bank,
    stats: () => ({ ...stats, stepsDone }),
    plant: (i = 0, x = G / 2, y = G / 2, r = 20, n = 400) => queuePlant(bank.entries[i], x, y, r, n),
  };
  const tStart = performance.now();
  device.queue.onSubmittedWorkDone().then(() => {});
  requestAnimationFrame(t => { frame(t); device.queue.onSubmittedWorkDone().then(() => console.log('first frame done in', Math.round(performance.now() - tStart), 'ms')); });
}

main().catch(e => fail(String(e?.stack || e)));
