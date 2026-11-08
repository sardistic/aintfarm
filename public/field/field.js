// Field Life — every particle is a unique, untrained, mutating neural net with mutable physics.
// Particles never interact directly: they read, absorb from and write to a G×G×(F+1) field tensor
// (F evolvable channels + a density channel). All simulation state lives on the GPU (WebGPU compute).
// A CPU-side gene bank archives evolved genomes as phenotype vectors (nearest-neighbour niches).

import { EFFECTS, layout, makeGenomeTools, asF32, genomeToB64, b64ToGenome, strainTagOf, strainHue, makeFounder } from './genebank.js';

const qs = new URLSearchParams(location.search);
const F = clampInt(qs.get('f'), 4, 16, 8);          // evolvable field channels (ch0 = energy)
const HMAX = clampInt(qs.get('h'), 2, 32, 12);      // max hidden width (actual width is a gene)
const G = clampInt(qs.get('g'), 128, 2048, 768);    // field grid size
const N_FIXED = qs.has('n') ? clampInt(qs.get('n'), 1024, 2_000_000, 262144) : 0;   // explicit size disables auto-fit
const SOURCES = 12;
const L = layout(F, HMAX);
const { FS, FX, KE, NT, NI, NO, P, HDR, EOFF, WOFF, WWORDS, GS } = L;
const GATHER_K = 64;             // genomes sampled per gene-bank pass
const PLANT_MAX = 600;
const PBYTES = 48;               // Particle struct size
if ((NI * HMAX + HMAX) % 2 || NO % 2) throw new Error('plasticity expects W2 and NO to be even');

function clampInt(v, lo, hi, d) { const n = parseInt(v, 10); return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d; }
const $ = id => document.getElementById(id);
function noWebGPU() {
  document.body.classList.add('nogpu');
  $('nogpu').hidden = false;
}
function fail(msg) { console.error(msg); const e = $('err'); e.textContent = msg; e.style.display = 'grid'; }

const common = (N) => /* wgsl */`
const F: u32 = ${F}u;
const FS: u32 = ${FS}u;
const FX: u32 = ${FX}u;
const SHARE: u32 = ${F + 2}u;      // per-cell energy left by bonded kin; only the cell's occupant can collect it
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

struct Particle { pos: vec2f, vel: vec2f, energy: f32, age: f32, alive: u32, kids: u32, mem: vec2f, rbar: f32, pad: f32 };

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
  plantPos: vec2f, cohesion: f32, learnOn: u32,
  effectMask: u32, founderBias: u32, myStrain: u32, strainFrac: f32,
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
@group(0) @binding(8) var<storage, read_write> counters: array<atomic<u32>, 48>;
// counters: 0 split requests, 1 free slots, 2 alive, 3 births (total), 4 deaths (total),
//           5 sum active params, 6 sum hidden width, 7 max generation, 8 step counter, 9 lineages seeded,
//           11 gene uptakes (total), 12 gene injections (total), 13 bonded particles, 14 learners,
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
fn dep(cell: u32, c: u32, v: f32) { atomicAdd(&deposit[cell * FX + c], i32(round(v * DSCALE))); }
fn chanValue(c: u32) -> f32 { return select(1.0 / INFO_GAIN, 1.0, c == 0u); } // energy per unit in channel c
fn activeParams(h: u32) -> u32 { return NI * h + h + h * NO + NO; }
fn stepSeed(salt: u32) -> u32 { return pcg(atomicLoad(&counters[8]) * 2654435761u + salt); }
fn packEffect(t: u32, a: u32, b: u32) -> u32 { return t | (a << 8u) | (b << 16u); }
// who occupied this cell last step (slot + 1, 0 = nobody); written via atomicMax, carried in the field's last slot
fn gradMag(c: u32, x: i32, y: i32) -> f32 {   // |∇| of channel c at a cell (L1, central differences)
  return abs(field[cellIdx(x + 1, y) * FX + c] - field[cellIdx(x - 1, y) * FX + c]) + abs(field[cellIdx(x, y + 1) * FX + c] - field[cellIdx(x, y - 1) * FX + c]);
}
fn occupant(cell: u32) -> u32 { return bitcast<u32>(field[cell * FX + FS]); }
fn copyEffect(srcSlot: u32, srcGene: u32, dstSlot: u32, dstGene: u32) {
  for (var w = 0u; w < 4u; w++) { sw(dstSlot, EOFF + dstGene * 4u + w, gw(srcSlot, EOFF + srcGene * 4u + w)); }
}

@compute @workgroup_size(1)
fn reset() {
  atomicStore(&counters[0], 0u);
  atomicStore(&counters[1], 0u);
  atomicStore(&counters[2], 0u);
  atomicStore(&counters[5], 0u);
  atomicStore(&counters[6], 0u);
  atomicStore(&counters[7], 0u);
  atomicStore(&counters[13], 0u);
  atomicStore(&counters[14], 0u);
  atomicStore(&counters[15], 0u);
  atomicStore(&counters[10], 0u);
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
  sf(b, 8u, select(0.0, rnd() * 0.05, rnd() < 0.5)); // within-life learning rate (0 = innate only)
  sw(b, 9u, pcg(lineage ^ 0x5bd1e995u));     // bond key (kin recognition)
  for (var c = 0u; c < F; c++) {
    sf(b, HDR + c, select(rnd() * 0.12, 0.02 + rnd() * 0.1, c == 0u));          // absorb rate
    sf(b, HDR + F + c, select(rndn() * 0.5, 0.5 + rnd() * 0.5, c == 0u));       // energy per unit absorbed
    sf(b, HDR + 2u * F + c, rndn() * 1.5);                                      // push along gradient
  }
  for (var s = 0u; s < KE; s++) {
    var t = select(0u, 1u + min(u32(rnd() * f32(NT - 1u)), NT - 2u), rnd() < 0.5);
    if (u.founderBias == 1u && s < 2u && rnd() < 0.7) { t = select(16u, 17u, s == 0u); }   // membrane founders: edge-seeking + bond
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
  p.rbar = 0.0;
  p.pad = 0.0;
  return p;
}

@compute @workgroup_size(128)
fn seed(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= N) { return; }
  if (parts[i].alive == 1u) { return; }
  rs = stepSeed(i * 3u + 17u);
  if (rnd() >= u.seedProb) { return; }
  if (rnd() < u.strainFrac) {                    // this visitor's strain: mutated copies of their thumbprint genome
    for (var w = 0u; w < GS; w++) { sw(i, w, uploadWord(w)); }
    mutateCopy(i, i);
  } else {
    randomGenome(i, pcg(i + atomicAdd(&counters[9], 1u) * 977u));
  }
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
    let v = field[here * FX + c];
    let g = vec2f(field[cr * FX + c] - field[cl * FX + c], field[cu * FX + c] - field[cd * FX + c]) * inv;
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
  if (occupant(here) == i + 1u) {                // collect energy bonded kin left in my cell
    let sh = field[here * FX + SHARE];
    if (sh > 0.0) { e += sh; dep(here, SHARE, -sh); }
  }

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
    if (t == 0u || t >= NT || ((u.effectMask >> t) & 1u) == 1u) { continue; }
    let ca = min((tw >> 8u) & 255u, F);          // may be the density channel
    let cb = min((tw >> 16u) & 255u, F - 1u);
    let cs = min(ca, F - 1u);                     // ca restricted to real channels
    let p0 = gf(b, base + 1u);
    let p1 = gf(b, base + 2u);
    let p2 = gf(b, base + 3u);
    let gate = sig(o[GATE0 + s]);
    // re-read from the (cache-hot) field rather than dynamically indexing private arrays, which spills them
    let vA = field[here * FX + ca];
    let vS = field[here * FX + cs];
    let ga = vec2f(field[cr * FX + ca] - field[cl * FX + ca], field[cu * FX + ca] - field[cd * FX + ca]) * inv;
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
        let amt = clamp(abs(p0), 0.0, 1.0) * gate * 0.15 * max(field[fc * FX + cs], 0.0);
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
      case 14u: {                                                             // uptake: copy a gene from whoever shares my cell
        if (rnd() < clamp(abs(p0), 0.0, 1.0) * gate * 0.02) {
          let d = occupant(here);
          if (d > 0u && d - 1u != i && parts[d - 1u].alive == 1u) {
            let j = d - 1u;
            let dst = (s + 1u + min(u32(rnd() * f32(KE - 1u)), KE - 2u)) % KE;   // never overwrite this uptake gene
            copyEffect(j, min(u32(rnd() * f32(KE)), KE - 1u), b, dst);
            if (rnd() < 0.3) {                                                // plus a block of 16 weight words
              let off = min(u32(rnd() * f32(WWORDS - 16u)), WWORDS - 16u);
              for (var w = 0u; w < 16u; w++) { sw(b, WOFF + off + w, gw(j, WOFF + off + w)); }
            }
            e -= 0.002;
            atomicAdd(&counters[11], 1u);
          }
        }
      }
      case 15u: {                                                             // inject: write this gene into a neighbour
        if (rnd() < clamp(abs(p0), 0.0, 1.0) * gate * 0.02) {
          let d = occupant(here);
          if (d > 0u && d - 1u != i && parts[d - 1u].alive == 1u) {
            copyEffect(b, s, d - 1u, min(u32(rnd() * f32(KE)), KE - 1u));
            e -= 0.01;
            atomicAdd(&counters[12], 1u);
          }
        }
      }
      case 16u: {                                                             // bond: spring to kin with a matching key
        let myKey = gw(b, 9u);
        let tol = u32(clamp(abs(p1) * 4.0, 0.0, 8.0));
        let rest = 0.8 + abs(p2);
        var bonded = false;
        for (var q = 0u; q < 5u; q++) {
          let ox = select(select(select(select(0, 1, q == 1u), -1, q == 2u), 0, q == 3u), 0, q == 4u);
          let oy = select(select(0, 1, q == 3u), -1, q == 4u);
          let d = occupant(cellIdx(x + ox, y + oy));
          if (d == 0u || d - 1u == i) { continue; }
          let j = d - 1u;
          if (parts[j].alive == 0u || countOneBits(myKey ^ gw(j, 9u)) > tol) { continue; }
          let dv = wrapD(parts[j].pos - p.pos);
          let dist = max(length(dv), 1e-3);
          force += (dv / dist) * (dist - rest) * abs(p0) * gate * 1.5;
          force += (parts[j].vel - p.vel) * gate * 0.8;
          let ej = parts[j].energy;
          if (e > ej * 1.2 + 0.05) {                 // share energy downhill to kin (conserved: it waits in their cell)
            let give = min((e - ej) * 0.05 * clamp(abs(p2), 0.0, 1.0) * gate, e * 0.05);
            e -= give;
            dep(cellIdx(i32(floor(parts[j].pos.x)), i32(floor(parts[j].pos.y))), SHARE, give);
            atomicAdd(&counters[15], u32(give * 100000.0));
          }
          bonded = true;
        }
        if (bonded) { atomicAdd(&counters[13], 1u); }
      }
      case 17u: {                                                             // edge: climb the steepness of channel a -> settle on boundaries
        let ge = vec2f(gradMag(ca, x + r, y) - gradMag(ca, x - r, y), gradMag(ca, x, y + r) - gradMag(ca, x, y - r));
        force += p0 * gate * ge / (length(ge) + 0.005);
      }
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
  let g1 = vec2f(field[cellIdx(x + 1, y) * FX + F] - field[cellIdx(x - 1, y) * FX + F],
                 field[cellIdx(x, y + 1) * FX + F] - field[cellIdx(x, y - 1) * FX + F]) * 0.5;
  force -= u.pressure * g1 / (length(g1) * 0.25 + 1.0);
  force += u.cohesion * gDen / (length(gDen) + 0.05);
  var v = p.vel * 0.85 + push * 0.25 + force * 0.12 + vec2f(rnd() - 0.5, rnd() - 0.5) * (0.04 + heat);
  v *= 1.0 - anchor * 0.9;
  let sp = length(v);
  if (sp > 1.2) { v = v * (1.2 / sp); }
  // --- learn: reward-modulated Hebbian update of this net's hidden->output weights (Lamarckian: children inherit them) ---
  let eta = gf(b, 8u);
  let dE = e - p.energy;
  if (u.learnOn == 1u && eta > 0.0005 && ((i + atomicLoad(&counters[8])) & 1u) == 0u) {
    let m = tanh((dE - p.rbar) * 40.0);
    let lr = eta * m;
    for (var h = 0u; h < H; h++) {
      let hh = lr * hid[h];
      for (var k = 0u; k < NO; k += 2u) {
        let wi = (WOFF + ((W2 + h * NO + k) >> 1u)) * N + b;
        let w = unpack2x16float(genome[wi]) + hh * vec2f(tanh(o[k]), tanh(o[k + 1u]));
        genome[wi] = pack2x16float(clamp(w, vec2f(-8.0), vec2f(8.0)));
      }
    }
  }
  p.rbar = p.rbar * 0.95 + dE * 0.05;

  p.vel = v;
  p.pos = wrapP(p.pos + v);
  p.energy = e;
  parts[i] = p;
  // occupancy for the density channel (at the new position)
  let nc = cellIdx(i32(floor(p.pos.x)), i32(floor(p.pos.y)));
  atomicAdd(&deposit[nc * FX + F], i32(DSCALE));
  atomicMax(&deposit[nc * FX + FS], i32(i + 1u));
}

var<workgroup> wsum: array<atomic<u32>, ${6 + NT}>;
@compute @workgroup_size(128)
fn collect(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li < ${6 + NT}u) { atomicStore(&wsum[li], 0u); }
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
      if (gf(i, 8u) > 0.0005) { atomicAdd(&wsum[4], 1u); }
      if (gw(i, 4u) == u.myStrain) { atomicAdd(&wsum[5], 1u); }
      for (var s = 0u; s < KE; s++) {
        let t = min(gw(i, EOFF + s * 4u) & 255u, NT - 1u);
        atomicAdd(&wsum[6u + t], 1u);
      }
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    atomicAdd(&counters[2], atomicLoad(&wsum[0]));
    atomicAdd(&counters[5], atomicLoad(&wsum[1]));
    atomicAdd(&counters[6], atomicLoad(&wsum[2]));
    atomicMax(&counters[7], atomicLoad(&wsum[3]));
    atomicAdd(&counters[14], atomicLoad(&wsum[4]));
    atomicAdd(&counters[10], atomicLoad(&wsum[5]));
  }
  if (li >= 6u && li < ${6 + NT}u) { atomicAdd(&counters[10u + li], atomicLoad(&wsum[li])); }
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
  if (rnd() < 0.002 * u.mutScale) { hue = rnd(); }   // occasional colour jump; the lineage word (strain tag) is strictly inherited
  sf(db, 3u, fract(hue + 1.0));
  sw(db, 4u, lineage);
  sw(db, 5u, gw(sb, 5u) + 1u);
  sf(db, 6u, clamp(gf(sb, 6u) * exp(rndn() * 0.05 * u.mutScale), 150.0, 8000.0));
  sf(db, 7u, clamp(gf(sb, 7u) * exp(rndn() * m), 0.002, 1.0));
  var eta = gf(sb, 8u);
  if (rnd() < 0.02 * u.mutScale) { eta = select(rnd() * 0.03, 0.0, eta > 0.0); } else { eta = eta * exp(rndn() * m); }
  sf(db, 8u, clamp(eta, 0.0, 0.2));
  var key = gw(sb, 9u);
  if (rnd() < 0.02 * u.mutScale) { key = key ^ (1u << min(u32(rnd() * 32.0), 31u)); }
  sw(db, 9u, key);
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
  if (j >= u.plantCount) { return; }
  rs = stepSeed(j * 2971u + 99991u);
  // use slots freed by the previous step first; when the dish is full, replace random particles elsewhere
  var slot = 0u;
  if (j < avail) { slot = freeList[used + j]; } else { slot = min(u32(rnd() * f32(N)), N - 1u); }
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
  fieldOut[id * FX + FS] = bitcast<f32>(u32(atomicExchange(&deposit[id * FX + FS], 0)));   // occupant id
  let shTot = max(field[id * FX + SHARE] + f32(atomicExchange(&deposit[id * FX + SHARE], 0)) / DSCALE, 0.0);
  fieldOut[id * FX + SHARE] = shTot * 0.9;
  let shareLeak = shTot * 0.1;                  // unclaimed kin energy returns to the commons
  for (var c = 0u; c < FS; c++) {
    let v = field[id * FX + c];
    let lap = field[l * FX + c] + field[r * FX + c] + field[dn * FX + c] + field[up * FX + c] - 4.0 * v;
    let d = f32(atomicExchange(&deposit[id * FX + c], 0)) / DSCALE;
    if (c == F) {                          // density: current occupancy, lightly smoothed
      fieldOut[id * FX + c] = clamp(d * 0.6 + (v + 0.2 * lap) * 0.4, 0.0, 64.0);
      continue;
    }
    let fc = f32(c) / f32(F);
    let diff = select(0.03 + 0.17 * fract(fc * 2.618 + 0.3), 0.04, c == 0u);
    let decay = select(0.004 + 0.03 * fract(fc * 1.618 + 0.1), 0.0006, c == 0u);
    var nv = v + diff * lap - decay * v + d;
    if (c == 0u) { nv += sun * u.sun + shareLeak; }
    fieldOut[id * FX + c] = clamp(nv, 0.0, 4.0);
  }
}
`;

const renderSrc = (N) => common(N) + /* wgsl */`
struct Cam { center: vec2f, scale: f32, mode: u32, screen: vec2f, pSize: f32, gain: f32, myStrain: u32, highlight: u32, style: u32, pad: u32 };
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
fn fval(cell: u32, c: u32) -> f32 { return field[cell * FX + c]; }

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

struct VO { @builtin(position) pos: vec4f, @location(0) col: vec3f, @location(1) uv: vec2f, @location(2) @interpolate(flat) shape: u32 };
// body plan from the dominant effect gene: 0 blob, 1 comet (movers), 2 ring (bonders), 3 diamond (anchor/shield),
// 4 star (gene uptake/injection), 5 cell with nucleus (learners), 6 dash (edge/isoline membrane builders)
fn shapeOf(t: u32, learner: bool) -> u32 {
  var r = select(0u, 5u, learner);
  switch t {
    case 1u, 2u, 5u, 13u: { r = 1u; }
    case 16u: { r = 2u; }
    case 6u, 12u: { r = 3u; }
    case 14u, 15u: { r = 4u; }
    case 3u, 17u: { r = 6u; }
    default: {}
  }
  return r;
}
@vertex fn vsP(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO;
  let p = parts[ii];
  if (p.alive == 0u || cam.mode >= 2u) { o.pos = vec4f(4.0, 4.0, 0.0, 1.0); return o; }
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                                vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let cn = corners[vi];
  // dominant effect gene -> body plan; moving particles stretch along their velocity
  var domT = 0u; var domW = 0.0;
  for (var k = 0u; k < KE; k++) {
    let t = genome[(EOFF + k * 4u) * N + ii] & 255u;
    let w = abs(bitcast<f32>(genome[(EOFF + k * 4u + 1u) * N + ii]));
    if (t > 0u && w > domW) { domW = w; domT = t; }
  }
  var shape = shapeOf(domT, bitcast<f32>(genome[8u * N + ii]) > 0.0005);
  let sp = length(p.vel);
  var dir = select(vec2f(1.0, 0.0), p.vel / max(sp, 1e-4), sp > 1e-3);
  var stretch = 1.0 + min(sp * 1.2, 0.8);
  var size = cam.pSize;
  if (cam.style == 1u) { shape = 0u; stretch = 1.0; }
  if (shape == 1u) { stretch = 1.4 + min(sp * 3.0, 2.2); }
  if (shape == 6u) { dir = vec2f(-dir.y, dir.x); stretch = 2.2; }
  if (shape == 2u || shape == 5u) { size *= 1.35; stretch = 1.0; }
  let perp = vec2f(-dir.y, dir.x);
  let d = wrapD(p.pos - cam.center);
  let s = d * cam.scale + (dir * cn.x * stretch + perp * cn.y) * size;
  o.pos = vec4f(s.x / (cam.screen.x * 0.5), -s.y / (cam.screen.y * 0.5), 0.0, 1.0);
  o.shape = shape;
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
  if (cam.highlight == 1u) { o.col *= select(0.1, 1.5, genome[4u * N + ii] == cam.myStrain); }
  o.uv = cn;
  return o;
}
@fragment fn fsP(in: VO) -> @location(0) vec4f {
  let q = in.uv;
  let r = length(q);
  var a = 0.0;
  switch in.shape {
    case 1u: {                                     // comet: bright head at +x, tapering tail
      let t = (q.x + 1.0) * 0.5;
      let w = mix(0.1, 0.95, t * t);
      a = (1.0 - smoothstep(w * 0.5, w, abs(q.y))) * (1.0 - smoothstep(0.8, 1.0, q.x)) * smoothstep(-1.0, -0.3, q.x);
    }
    case 2u: { a = 1.0 - smoothstep(0.1, 0.26, abs(r - 0.66)); }                     // ring
    case 3u: { a = 1.0 - smoothstep(0.5, 0.95, abs(q.x) + abs(q.y)); }               // diamond
    case 4u: {                                     // star / cross
      let m = min(abs(q.x), abs(q.y));
      a = max((1.0 - smoothstep(0.08, 0.28, m)) * (1.0 - smoothstep(0.6, 1.0, r)), 1.0 - smoothstep(0.1, 0.32, r));
    }
    case 5u: {                                     // cell: membrane, faint cytoplasm, nucleus
      a = max(0.9 * (1.0 - smoothstep(0.07, 0.2, abs(r - 0.72))), 1.0 - smoothstep(0.16, 0.32, r)) + 0.15 * (1.0 - smoothstep(0.62, 0.72, r));
    }
    case 6u: { a = (1.0 - smoothstep(0.2, 0.6, abs(q.y))) * (1.0 - smoothstep(0.65, 1.0, abs(q.x))); }   // dash
    default: { a = 1.0 - smoothstep(0.35, 1.0, r); }
  }
  a = clamp(a, 0.0, 1.0);
  return vec4f(in.col * a, a);
}
`;

// Gene bank, genome decoding and embedding live in genebank.js, shared with the server.
const { decodeGenome, validateGenome, embed, phenoColor, weights, GeneBank } = makeGenomeTools(L);

async function main() {
  if (!navigator.gpu) return noWebGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No WebGPU adapter found.');
  const lim = adapter.limits;
  const maxBind = Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize);
  // Auto-fit: start from a device guess (or what fit last time), then halve and reload if frame rate is poor.
  const ainfo0 = adapter.info || {};
  const guess = ainfo0.isFallbackAdapter ? 16384 : (matchMedia('(pointer: coarse)').matches && innerWidth < 900 ? 65536 : 262144);
  let stored = 0; try { stored = parseInt(localStorage.getItem('fieldlife-n'), 10) || 0; } catch {}
  const N_GUESS = guess;
  const N_REQ = N_FIXED || Math.min(guess, stored || guess);
  let N = Math.min(N_REQ, Math.floor(maxBind / (GS * 4)));
  N = Math.floor(N / 128) * 128;
  if (G * G * FX * 4 > maxBind) return fail(`Field ${G}×${G}×${FX} exceeds this GPU's buffer limit.`);
  const ainfo = adapter.info || {};
  console.log('adapter', ainfo.vendor, ainfo.architecture, ainfo.description, 'N', N, 'GS', GS, 'P', P);
  const hasTS = adapter.features.has('timestamp-query');
  // WebGPU only exposes vendor/architecture (browsers hide the exact model); throughput is measured instead
  const gpuLabel = [ainfo0.vendor, ainfo0.architecture].filter(Boolean).join(' ') || 'unknown GPU';
  const device = await adapter.requestDevice({
    requiredFeatures: hasTS ? ['timestamp-query'] : [],
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
  const UBYTES = 80 + SOURCES * 16;
  const uni = buf(UBYTES, GPUBufferUsage.UNIFORM | CD);
  const camBuf = buf(48, GPUBufferUsage.UNIFORM | CD);
  const parts = buf(N * PBYTES, S | CD | CS);
  const genome = buf(N * GS * 4, S | CD);
  const fieldBufs = [buf(G * G * FX * 4, S | CD | CS), buf(G * G * FX * 4, S | CD | CS)];
  const deposit = buf(G * G * FX * 4, S | CD);
  const reqs = buf(Math.max(N * 8, GS * 4 + 16), S | CD);
  const freeList = buf(Math.max(N * 4, (GATHER_K * (GS + 4)) * 4), S | CS);
  const counters = buf(192, S | CS | CD);
  const tsSet = hasTS ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const tsResolve = hasTS ? buf(16, GPUBufferUsage.QUERY_RESOLVE | CS) : null;
  const tsRead = hasTS ? buf(16, GPUBufferUsage.MAP_READ | CD) : null;
  let tsPending = false, simMs = 0;
  const staging = buf(192, GPUBufferUsage.MAP_READ | CD);
  const GATHER_BYTES = GATHER_K * (GS + 4) * 4;
  const PICK_R = 3, PICK_W = 2 * PICK_R + 1, PICK_BYTES = PICK_W * PICK_W * FX * 4;
  const pickRead = buf(PICK_BYTES, GPUBufferUsage.MAP_READ | CD);
  const PROBE_G = (GS + 4) * 4, PROBE_P = PROBE_G, PROBE_F = PROBE_P + PBYTES, PROBE_BYTES = PROBE_F + 5 * FX * 4;
  const probeRead = buf(PROBE_BYTES, GPUBufferUsage.MAP_READ | CD);
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
  for (const id of ['sun', 'metab', 'pcost', 'mut', 'pressure', 'cohesion', 'spf', 'brush', 'fgain', 'commit']) {
    const el = $(id), out = el.nextElementSibling;
    if (qs.get(id)) el.value = qs.get(id);
    const sync = () => { ui[id] = parseFloat(el.value); out.textContent = fmtSlider(id, ui[id]); };
    el.addEventListener('input', sync); sync();
  }
  function fmtSlider(id, v) {
    if (id === 'metab') return (v * 1000).toFixed(1) + 'e-3';
    if (id === 'pcost') return (v * 1e6).toFixed(1) + 'e-6';
    if (id === 'commit') return v + '%';
    return String(v);
  }
  let mode = clampInt(qs.get('mode'), 0, 4, 0), paused = false, cur = 0, seedProb = 0.35, frameNo = 0;
  let tool = 'erase';
  const cam = { cx: G / 2, cy: G / 2, scale: 1 };
  const mouse = { x: 0, y: 0, down: false, panning: false, lx: 0, ly: 0 };
  const stats = { mine: 0, alive: 0, births: 0, deaths: 0, params: 0, hidden: 0, gen: 0, uptakes: 0, injections: 0, bonded: 0, learners: 0, shared: 0, effects: new Array(NT).fill(0), pending: false };
  let commitAcc = 0;
  let stepsDone = 0, stepRate = 0, fps = 0, rateT = performance.now(), rateSteps = 0, rateFrames = 0;
  const bank = new GeneBank();
  const BANK_KEY = `fieldlife-bank-v4-F${F}-H${HMAX}`;
  try { const saved = localStorage.getItem(BANK_KEY); if (saved && !qs.has('fresh')) bank.load(saved); } catch (e) { console.warn('gene bank not restored', e); }
  // ---- your strain: a stable thumbprint per browser ----
  let visitorId = '';
  try { visitorId = localStorage.getItem('fieldlife-visitor') || ''; } catch {}
  if (!/^[0-9a-f]{32}$/.test(visitorId)) {
    visitorId = Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('');
    try { localStorage.setItem('fieldlife-visitor', visitorId); } catch {}
  }
  const strainTag = strainTagOf(visitorId);
  const strainHex = strainTag.toString(16).padStart(8, '0');
  const STRAIN_KEY = `fieldlife-strain-v1-F${F}-H${HMAX}`;
  let strainBest = { g: makeFounder(L, strainTag), score: -1, gen: 0 };
  try {
    const st = JSON.parse(localStorage.getItem(STRAIN_KEY) || 'null');
    const g = st && b64ToGenome(st.g);
    if (g && !validateGenome(g) && (g[4] >>> 0) === strainTag) strainBest = { g, score: st.score || 0, gen: g[5] };
  } catch {}
  // Your strain is your GPU's champion: the most successful net your dish has produced (offspring, age, generation).
  // When a better one appears it takes over your thumbprint (lineage tag + hue), so the strain keeps advancing while you run.
  // `local` marks genomes sampled from your own dish; shared-bank genomes only count if they already carry your tag.
  let championChanges = 0;
  function considerOwn(g, fitness, local = false) {
    const own = (g[4] >>> 0) === strainTag;
    if ((!own && !local) || validateGenome(g)) return;
    const score = fitness + g[5] * 0.05;
    if (score <= strainBest.score) return;
    const t = Uint32Array.from(g);
    t[4] = strainTag;
    const dv = new DataView(new ArrayBuffer(4)); dv.setFloat32(0, strainHue(strainTag), true); t[3] = dv.getUint32(0, true);
    strainBest = { g: t, score, gen: t[5] };
    if (!own) championChanges++;
    try { localStorage.setItem(STRAIN_KEY, JSON.stringify({ g: genomeToB64(t), score })); } catch {}
  }
  for (const e of bank.entries) considerOwn(e.genome, e.fitness);
  // A brand-new strain is founded from the fittest genome already known, re-tagged with your thumbprint,
  // so it starts viable and diverges from there (a random founder almost always goes extinct).
  function adoptFounder() {
    if (strainBest.score >= 0 || !bank.entries.length) return false;
    const best = bank.entries.reduce((a, b) => (b.fitness > a.fitness ? b : a));
    const g = Uint32Array.from(best.genome);
    g[4] = strainTag; g[5] = 0;
    const dv = new DataView(new ArrayBuffer(4)); dv.setFloat32(0, strainHue(strainTag), true); g[3] = dv.getUint32(0, true);
    strainBest = { g, score: 0, gen: 0 };
    try { localStorage.setItem(STRAIN_KEY, JSON.stringify({ g: genomeToB64(g), score: 0 })); } catch {}
    return true;
  }
  adoptFounder();
  let seedStrainFrac = 0.4, lastStrainPlant = performance.now(), plantedChampion = 0;
  const WORK_KEY = `fieldlife-work-${strainHex}`;
  let lifeWork = 0; try { lifeWork = +(localStorage.getItem(WORK_KEY) || 0); } catch {}
  // your node: where your strain enters every dish. Random until you choose it with the node tool.
  let myNode = null;
  try { myNode = JSON.parse(localStorage.getItem('fieldlife-node') || 'null'); } catch {}
  if (!myNode || !(myNode.x >= 0 && myNode.x < 1 && myNode.y >= 0 && myNode.y < 1)) myNode = { x: Math.random(), y: Math.random(), chosen: false };
  const saveNode = () => { try { localStorage.setItem('fieldlife-node', JSON.stringify(myNode)); } catch {} };
  saveNode();
  let nodeDirty = true;
  function drawThumb(canvasEl, tag) {
    const c = canvasEl.getContext('2d'), n = 5, w = canvasEl.width / n;
    let bits = strainTagOf('thumb:' + tag);
    c.clearRect(0, 0, canvasEl.width, canvasEl.height);
    c.fillStyle = `hsl(${(strainHue(tag) * 360).toFixed(0)} 85% 60%)`;
    for (let y = 0; y < n; y++) for (let x = 0; x < 3; x++) {
      if ((bits >>> (y * 3 + x)) & 1) { c.fillRect(x * w, y * w, w, w); c.fillRect((n - 1 - x) * w, y * w, w, w); }
    }
  }

  // ---- particle inspector ----
  const insp = { slot: -1, pick: null, busy: false, last: 0, pos: [0, 0], r: 1, data: null };
  const bootT = performance.now();
  let fitSamples = [], fitDone = !!N_FIXED;
  let fitNote = '';
  try { const dn = sessionStorage.getItem('fieldlife-downsized'); if (dn) { fitNote = `fitted to ${(+dn).toLocaleString()} particles for this GPU`; sessionStorage.removeItem('fieldlife-downsized'); } } catch {}

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
    if (tool === 'node') {
      myNode = { x: (((mouse.x % G) + G) % G) / G, y: (((mouse.y % G) + G) % G) / G, chosen: true };
      saveNode(); nodeDirty = true; lastUpload = 0;
      queuePlant({ genome: strainBest.g }, myNode.x * G, myNode.y * G, 16, 450);
      setTool('erase');
      return;
    }
    if (tool === 'inspect') { insp.pick = [((mouse.x % G) + G) % G, ((mouse.y % G) + G) % G]; return; }
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
  if (qs.get('learn') === '0') $('learn').checked = false;
  // ?off=bond,inject disables effect types (for A/B comparisons and profiling)
  let effectMask = (qs.get('off') || '').split(',').reduce((m, n) => { const t = EFFECTS.indexOf(n.trim()); return t > 0 ? m | (1 << t) : m; }, 0);
  let founderBias = qs.get('preset') === 'membranes' ? 1 : 0;
  const PRESETS = {
    default: { pressure: 0.3, cohesion: 0.15, bias: 0 },
    membranes: { pressure: 0.55, cohesion: 0.4, bias: 1 },
    soup: { pressure: 0.1, cohesion: 0.0, bias: 0 },
  };
  function applyPreset(name, resetToo) {
    const pr = PRESETS[name]; if (!pr) return;
    for (const id of ['pressure', 'cohesion']) { $(id).value = pr[id]; $(id).dispatchEvent(new Event('input')); }
    founderBias = pr.bias;
    if (resetToo) resetWorld();
  }
  $('preset').addEventListener('change', () => applyPreset($('preset').value, true));
  if (PRESETS[qs.get('preset')]) { $('preset').value = qs.get('preset'); applyPreset(qs.get('preset'), false); }
  if (innerWidth < 700) { $('panel').classList.add('collapsed'); $('collapse').textContent = '+'; }
  const modeSel = $('mode');
  modeSel.value = String(mode);
  modeSel.addEventListener('change', () => { mode = +modeSel.value; });
  const pauseBtn = $('pause');
  const togglePause = () => { paused = !paused; pauseBtn.textContent = paused ? 'play' : 'pause'; };
  pauseBtn.addEventListener('click', togglePause);
  $('seed').addEventListener('click', () => { seedProb = 0.1; seedStrainFrac = 0; });
  $('reset').addEventListener('click', resetWorld);
  $('collapse').addEventListener('click', () => {
    const p = $('panel'); p.classList.toggle('collapsed');
    $('collapse').textContent = p.classList.contains('collapsed') ? '+' : '–';
  });
  const setTool = t => { tool = t; for (const [id, n] of [['toolErase', 'erase'], ['toolPlant', 'plant'], ['toolInspect', 'inspect'], ['toolNode', 'node']]) $(id).classList.toggle('on', t === n); };
  $('toolInspect').addEventListener('click', () => setTool('inspect'));
  $('toolNode').addEventListener('click', () => setTool('node'));
  $('inspClose').addEventListener('click', () => { insp.slot = -1; insp.data = null; $('inspector').hidden = true; $('ring').hidden = true; });
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
    if (e.key === 'i') setTool('inspect');
    if (e.key === 'n') setTool('node');
  });
  function saveBank() {
    try { localStorage.setItem(BANK_KEY, bank.serialize()); } catch (e) { console.warn('gene bank not saved', e); }
    try { localStorage.setItem(WORK_KEY, String(Math.round(lifeWork))); } catch {}
  }
  addEventListener('beforeunload', saveBank);

  function plantRandom() {
    if (!bank.entries.length) return;
    // every niche gets a base chance; niches of strains whose visitors have contributed more GPU work get extra
    const w = bank.entries.map(e => 1 + (strainWeight.get(e.dec.lineage) || 0));
    let r = Math.random() * w.reduce((a, b) => a + b, 0), k = 0;
    while (k < w.length - 1 && (r -= w[k]) > 0) k++;
    const nd = strainNode.get(bank.entries[k].dec.lineage);   // a visitor's strain always enters at its node
    queuePlant(bank.entries[k], nd ? nd.x * G : Math.random() * G, nd ? nd.y * G : Math.random() * G, 14, 360);
  }

  function resetWorld() {
    const enc = device.createCommandEncoder();
    enc.clearBuffer(parts); enc.clearBuffer(fieldBufs[0]); enc.clearBuffer(fieldBufs[1]);
    enc.clearBuffer(deposit); enc.clearBuffer(counters);
    device.queue.submit([enc.finish()]);
    seedProb = 0.35;
    seedStrainFrac = 0.4;
    plantQ.length = 0;
  }

  // Recomputes this particle's forward pass from its genome and the field values it sensed, then draws it.
  const OUT_LABELS = () => ['split', 'share', 'write', 'amount', ...Array.from({ length: F }, (_, c) => c ? 'aim c' + c : 'aim E'), 'mem 0', 'mem 1'];
  function renderInspector() {
    const D = insp.data; if (!D) return;
    const dec = decodeGenome(D.g), w = weights(D.g), H = dec.H;
    if (!D.alive) { $('insp').textContent = 'this particle has died. Click another one.'; insp.slot = -1; $('ring').hidden = true; return; }
    const [here, xp, xm, yp, ym] = D.cells, r = insp.r, inv = 1 / (2 * r);
    const inp = new Float32Array(NI);
    for (let c = 0; c < FS; c++) {
      const k = c === F ? 0.3 : 1;
      inp[c] = here[c] * k; inp[FS + c] = (xp[c] - xm[c]) * inv * 8 * k; inp[2 * FS + c] = (yp[c] - ym[c]) * inv * 8 * k;
    }
    inp[3 * FS] = D.energy - 1; inp[3 * FS + 1] = D.vel[0] * 2; inp[3 * FS + 2] = D.vel[1] * 2;
    inp[3 * FS + 3] = D.mem[0]; inp[3 * FS + 4] = D.mem[1]; inp[3 * FS + 5] = Math.sin(D.age * dec.clock);
    const hid = new Float32Array(H), out = new Float32Array(NO);
    for (let h = 0; h < H; h++) { let a = w[L.B1 + h]; for (let j = 0; j < NI; j++) a += w[j * HMAX + h] * inp[j]; hid[h] = Math.tanh(a); }
    for (let k = 0; k < NO; k++) { let a = w[L.B2 + k]; for (let h = 0; h < H; h++) a += w[L.W2 + h * NO + k] * hid[h]; out[k] = a; }
    const sig = x => 1 / (1 + Math.exp(-x));
    const labels = [...OUT_LABELS(), ...dec.effects.map(e => (EFFECTS[e.t] === 'none' ? '—' : EFFECTS[e.t]))];
    // diagram
    const cv = $('netc'), c = cv.getContext('2d'), W = cv.width, Hh = cv.height;
    c.fillStyle = '#050607'; c.fillRect(0, 0, W, Hh);
    const col = v => v >= 0 ? `rgba(247,190,70,${Math.min(1, Math.abs(v))})` : `rgba(90,200,255,${Math.min(1, Math.abs(v))})`;
    const ys = (n, i) => 10 + (Hh - 20) * (n === 1 ? 0.5 : i / (n - 1));
    const X0 = 18, X1 = W * 0.5, X2 = W - 92;
    c.lineWidth = 1;
    for (let h = 0; h < H; h++) {
      for (let j = 0; j < NI; j++) { const v = w[j * HMAX + h] * inp[j]; if (Math.abs(v) < 0.15) continue; c.strokeStyle = col(v * 0.6); c.beginPath(); c.moveTo(X0, ys(NI, j)); c.lineTo(X1, ys(H, h)); c.stroke(); }
      for (let k = 0; k < NO; k++) { const v = w[L.W2 + h * NO + k] * hid[h]; if (Math.abs(v) < 0.15) continue; c.strokeStyle = col(v * 0.6); c.beginPath(); c.moveTo(X1, ys(H, h)); c.lineTo(X2, ys(NO, k)); c.stroke(); }
    }
    const dot = (x, y, v, rr) => { c.fillStyle = col(Math.tanh(v) * 0.85 + Math.sign(v) * 0.15); c.beginPath(); c.arc(x, y, rr, 0, 6.2832); c.fill(); c.strokeStyle = 'rgba(255,255,255,.25)'; c.stroke(); };
    for (let j = 0; j < NI; j++) dot(X0, ys(NI, j), inp[j], 3);
    for (let h = 0; h < H; h++) dot(X1, ys(H, h), hid[h], 5);
    c.font = '10px ui-monospace, monospace'; c.textBaseline = 'middle';
    for (let k = 0; k < NO; k++) { dot(X2, ys(NO, k), out[k], 3.5); c.fillStyle = '#9fb3a8'; c.fillText(labels[k] || '', X2 + 7, ys(NO, k)); }
    c.fillStyle = '#7a8a80'; c.fillText(`${NI} senses`, 2, Hh - 4); c.fillText(`${H} hidden`, X1 - 22, Hh - 4);
    // text
    const gates = dec.effects.map((e, k) => e.t ? `${EFFECTS[e.t]} ${(100 * sig(out[6 + F + k])).toFixed(0)}%` : null).filter(Boolean);
    const splitting = out[0] > 0 && D.energy > 0.5 && D.age > 16;
    $('insp').textContent =
      `slot ${insp.slot}  gen ${dec.gen}  age ${D.age.toFixed(0)}  kids ${D.kids}\n` +
      `energy ${D.energy.toFixed(3)}  speed ${Math.hypot(...D.vel).toFixed(2)}\n` +
      `strain ${dec.lineage.toString(16).padStart(8, '0')}${dec.lineage === strainTag ? ' (yours)' : ''}  bond ${dec.key.toString(16).padStart(8, '0')}\n` +
      `net ${NI}→${H}→${NO}  ${dec.params} params  ${dec.eta > 0.0005 ? 'learning η ' + dec.eta.toFixed(3) : 'innate only'}\n` +
      `now: ${splitting ? 'SPLITTING (child gets ' + (15 + 70 * sig(out[1])).toFixed(0) + '%)' : 'not splitting'} · ${out[2] > 0 ? 'writing' : 'not writing'}\n` +
      `effect gates: ${gates.join(', ') || 'none'}`;
  }
  function placeRing() {
    if (insp.slot < 0 || !insp.data) { $('ring').hidden = true; return; }
    const dpr = canvas.width / innerWidth;
    const wrap = d => d - Math.round(d / G) * G;
    const sx = (wrap(insp.pos[0] - cam.cx) * cam.scale + canvas.width / 2) / dpr, sy = (wrap(insp.pos[1] - cam.cy) * cam.scale + canvas.height / 2) / dpr;
    const ring = $('ring'); ring.hidden = false;
    ring.style.transform = `translate(${sx - 14}px, ${sy - 14}px)`;
  }

  // ----- shared gene bank (one archive across all visitors) -----
  const SHARED = F === 8 && HMAX === 12 && !qs.has('offline');
  const sid = (() => {
    const mk = () => Array.from(crypto.getRandomValues(new Uint8Array(12)), b => (b % 36).toString(36)).join('');
    try { let v = sessionStorage.getItem('fieldlife-sid'); if (!v) { v = mk(); sessionStorage.setItem('fieldlife-sid', v); } return v; } catch { return mk(); }
  })();
  const strainWeight = new Map();      // lineage tag -> extra immigration weight from contributed GPU work
  const strainNode = new Map();        // lineage tag -> {x, y} in world fractions (shared by every visitor)
  let shared = null, sharedSince = 0, syncBusy = false, lastSync = -1e9, lastUpload = performance.now(), workPending = 0, firstImmig = true;
  async function syncShared() {
    if (!SHARED || syncBusy) return;
    syncBusy = true;
    try {
      const r = await fetch(`/api/bank?since=${sharedSince}&sid=${sid}&strain=${strainHex}`);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const o = await r.json();
      sharedSince = o.now;
      shared = o;
      strainWeight.clear();
      strainNode.clear();
      for (const st of o.strains || []) {
        const tag = parseInt(st.tag, 16) >>> 0;
        strainWeight.set(tag, Math.log10(1 + st.work / 1e7));
        if (st.node) strainNode.set(tag, st.node);
      }
      for (const e of o.entries) {
        const g = b64ToGenome(e.g);
        if (!g || validateGenome(g)) continue;
        const ins = bank.insert(g, e.fitness, e.kids, 'shared', e.t);
        if (ins) ins.dirty = false;
        considerOwn(g, e.fitness);
      }
      if (adoptFounder()) queuePlant({ genome: strainBest.g }, myNode.x * G, myNode.y * G, 16, 450);
      if (firstImmig && o.entries.length >= 8 && $('immigrate').checked) { firstImmig = false; for (let k = 0; k < 3; k++) plantRandom(); }
      drawBank();
    } catch (e) { console.warn('shared bank sync failed', e.message); }
    syncBusy = false;
  }
  async function uploadShared() {
    if (!SHARED) return;
    const picks = bank.entries.filter(e => e.dirty && e.origin === 'local').sort((a, b) => b.fitness - a.fitness).slice(0, 12);
    const body = { sid, strain: strainHex, work: Math.round(workPending), F, HMAX, GS, node: nodeDirty ? myNode : undefined,
      entries: picks.map(e => ({ g: genomeToB64(e.genome), fitness: e.fitness, kids: e.kids })) };
    try {
      const r = await fetch('/api/bank', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (r.ok) { picks.forEach(e => { e.dirty = false; }); workPending = 0; nodeDirty = false; }
    } catch (e) { console.warn('shared bank upload failed', e.message); }
  }
  const fmtWork = w => w >= 1e12 ? (w / 1e12).toFixed(1) + 'T' : w >= 1e9 ? (w / 1e9).toFixed(1) + 'B' : w >= 1e6 ? (w / 1e6).toFixed(1) + 'M' : Math.round(w / 1e3) + 'k';
  // world map: every strain's node, in the shared coordinate system all dishes use
  function nodeList() {
    const list = (shared?.strains || []).filter(st => st.node && st.tag !== strainHex)
      .map(st => ({ tag: st.tag, x: st.node.x, y: st.node.y, work: st.work, online: st.online, mine: false }));
    list.push({ tag: strainHex, x: myNode.x, y: myNode.y, work: Math.max(shared?.mine?.work || 0, lifeWork), online: true, mine: true });
    return list;
  }
  const world = $('world'), wctx = world.getContext('2d');
  function drawWorld() {
    const W = world.width;
    wctx.fillStyle = '#050607'; wctx.fillRect(0, 0, W, W);
    wctx.strokeStyle = 'rgba(255,255,255,.06)'; wctx.lineWidth = 1;
    for (let k = 1; k < 4; k++) { wctx.beginPath(); wctx.moveTo(k * W / 4, 0); wctx.lineTo(k * W / 4, W); wctx.moveTo(0, k * W / 4); wctx.lineTo(W, k * W / 4); wctx.stroke(); }
    // current viewport
    const vw = canvas.width / cam.scale / G * W, vh = canvas.height / cam.scale / G * W;
    wctx.strokeStyle = 'rgba(111,227,168,.5)'; wctx.strokeRect(cam.cx / G * W - vw / 2, cam.cy / G * W - vh / 2, vw, vh);
    for (const n of nodeList()) {
      const x = n.x * W, y = n.y * W, r = 4 + Math.min(14, Math.log10(1 + n.work / 1e6) * 3);
      wctx.globalAlpha = n.online ? 0.95 : 0.45;
      wctx.fillStyle = `hsl(${(strainHue(parseInt(n.tag, 16) >>> 0) * 360).toFixed(0)} 85% 60%)`;
      wctx.beginPath(); wctx.arc(x, y, r, 0, 6.2832); wctx.fill();
      if (n.mine) {
        wctx.globalAlpha = 1; wctx.strokeStyle = '#f7cf5a'; wctx.lineWidth = 3;
        wctx.beginPath(); wctx.arc(x, y, r + 5, 0, 6.2832); wctx.stroke();
        wctx.fillStyle = '#f7cf5a'; wctx.font = 'bold 22px ui-monospace, monospace'; wctx.fillText('you', x + r + 8, y + 7);
      }
    }
    wctx.globalAlpha = 1;
  }
  world.addEventListener('click', e => {
    const r = world.getBoundingClientRect();
    cam.cx = (e.clientX - r.left) / r.width * G; cam.cy = (e.clientY - r.top) / r.height * G;
  });
  // markers in the dish itself
  const nodeEls = new Map();
  function placeNodes() {
    const dpr = canvas.width / innerWidth, wrap = d => d - Math.round(d / G) * G, keep = new Set();
    for (const n of nodeList()) {
      keep.add(n.tag);
      let el = nodeEls.get(n.tag);
      if (!el) {
        el = document.createElement('div'); el.className = 'node' + (n.mine ? ' mine' : '');
        el.style.setProperty('--h', (strainHue(parseInt(n.tag, 16) >>> 0) * 360).toFixed(0));
        const label = document.createElement('span');
        label.textContent = n.mine ? 'your node' : n.tag.slice(0, 4) + '·' + n.tag.slice(4);
        el.appendChild(label);
        $('nodes').appendChild(el); nodeEls.set(n.tag, el);
      }
      const sx = (wrap(n.x * G - cam.cx) * cam.scale + canvas.width / 2) / dpr, sy = (wrap(n.y * G - cam.cy) * cam.scale + canvas.height / 2) / dpr;
      el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px)`;
      el.classList.toggle('off', !n.online);
    }
    for (const [tag, el] of nodeEls) if (!keep.has(tag)) { el.remove(); nodeEls.delete(tag); }
  }

  function renderStrains() {
    drawThumb($('thumb'), strainTag);
    const mine = shared?.mine;
    const share = stats.alive ? (100 * stats.mine / stats.alive) : 0;
    $('strainInfo').textContent =
      `strain     ${strainHex.slice(0, 4)}·${strainHex.slice(4)}\n` +
      `node       ${(myNode.x * 100).toFixed(0)},${(myNode.y * 100).toFixed(0)}${myNode.chosen ? ' (chosen)' : ' (random — use node tool)'}\n` +
      `in dish    ${share.toFixed(1)}% (${stats.mine.toLocaleString()})\n` +
      `advanced   gen ${Math.max(strainBest.gen, mine?.bestGen || 0)}\n` +
      `score      ${fmtWork(Math.max(mine?.work || 0, lifeWork))} particle-steps${mine ? ' · #' + ((shared.strains || []).findIndex(s => s.tag === strainHex) + 1 || '–') + ' of ' + shared.strainCount : ''}\n` +
      `GPU        ${gpuLabel} · ${fmtWork(stepRate * stats.alive)}/s at ${ui.commit}%`;
    const rows = (shared?.strains || []).slice(0, 10);
    $('strains').innerHTML = rows.length ? rows.map((st, i) => {
      const hue = (strainHue(parseInt(st.tag, 16) >>> 0) * 360).toFixed(0);
      const you = st.tag === strainHex ? ' <b>you</b>' : '';
      return `<div class="srow"><span class="sw" style="background:hsl(${hue} 85% 60%)"></span>${i + 1}. ${st.tag.slice(0, 4)}·${st.tag.slice(4)}${you}` +
        `<span class="sr">${fmtWork(st.work)} · gen ${st.bestGen} · ${st.niches}n${st.online ? ' ●' : ''}</span></div>`;
    }).join('') : '<div class="dim">no shared strains yet</div>';
    $('online').textContent = shared ? `${shared.online} online · ${shared.niches} shared niches · ${shared.contributors} contributors` : (SHARED ? 'connecting…' : 'offline (custom configuration)');
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
        if (e.dec.lineage === strainTag) { mctx.globalAlpha = 1; mctx.strokeStyle = '#f7cf5a'; mctx.lineWidth = 2; mctx.beginPath(); mctx.arc(x, y, r + 2.5, 0, 6.2832); mctx.stroke(); }
        else if (e.origin === 'shared') { mctx.globalAlpha = 0.6; mctx.strokeStyle = '#cfe8ff'; mctx.lineWidth = 1; mctx.beginPath(); mctx.arc(x, y, r + 1.5, 0, 6.2832); mctx.stroke(); }
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
      const two = ['catalyse', 'relay'].includes(n) ? `${ch(Math.min(e.a, F - 1))}→${ch(e.b)}` : ['pulse'].includes(n) ? `→${ch(e.b)}` : ['jet', 'anchor', 'heat', 'shield', 'drift', 'uptake', 'inject', 'bond'].includes(n) ? '' : ch(e.a);
      return `${n}${two ? '(' + two + ')' : ''} ${e.p0 >= 0 ? '+' : ''}${e.p0.toFixed(2)}`;
    });
    const eats = d.chem.egain.map((g, c) => [g * d.chem.absorb[c], c]).sort((a, b) => b[0] - a[0]);
    $('species').textContent =
      `#${selected.id}  gen ${d.gen}  kids ${selected.kids}  fit ${selected.fitness.toFixed(1)}\n` +
      `strain ${d.lineage.toString(16).padStart(8, '0')}${d.lineage === strainTag ? ' (yours)' : ''} · ${selected.origin === 'shared' ? 'from the shared bank' : 'evolved here'}\n` +
      `net ${NI}→${d.H}→${NO} (${d.params} params)  range ${d.range.toFixed(1)}\n` +
      `learning ${d.eta > 0.0005 ? 'η ' + d.eta.toFixed(3) : 'innate only'}  bond key ${d.key.toString(16).padStart(8, '0')}\n` +
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
  const camData = new ArrayBuffer(48);
  const camF = new Float32Array(camData), camU = new Uint32Array(camData);
  const wg = Math.ceil(N / 128), fg = Math.ceil(G / 16);

  function writeUniforms(extra = {}) {
    uF[0] = seedProb; uF[1] = ui.sun; uU[2] = extra.mouseOn ?? 0; uF[3] = ui.metab;
    uF[4] = extra.mx ?? 0; uF[5] = extra.my ?? 0; uF[6] = extra.mr ?? 0; uF[7] = ui.pcost;
    uF[8] = ui.mut; uF[9] = ui.pressure; uU[10] = extra.count ?? 0; uF[11] = extra.pr ?? 0;
    uF[12] = extra.px ?? 0; uF[13] = extra.py ?? 0; uF[14] = ui.cohesion; uU[15] = $('learn').checked ? 1 : 0;
    uU[16] = effectMask; uU[17] = founderBias; uU[18] = strainTag; uF[19] = extra.strainFrac ?? 0;
    sources.forEach((s, k) => {
      uF[20 + k * 4] = s.x; uF[21 + k * 4] = s.y; uF[22 + k * 4] = s.sigma;
      uF[23 + k * 4] = s.base * (0.35 + 0.65 * (0.5 + 0.5 * Math.sin(s.phase)));
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
    // inspector: find who is under the click (occupant ids in a small window of the field)
    if (insp.pick && !insp.busy) {
      const [px, py] = insp.pick; insp.pick = null; insp.busy = true;
      const cx = Math.floor(px), cy = Math.floor(py), x0 = Math.min(Math.max(cx - PICK_R, 0), G - PICK_W);
      const enc = device.createCommandEncoder();
      for (let k = 0; k < PICK_W; k++) {
        const y = (((cy - PICK_R + k) % G) + G) % G;
        enc.copyBufferToBuffer(fieldBufs[cur], (y * G + x0) * FX * 4, pickRead, k * PICK_W * FX * 4, PICK_W * FX * 4);
      }
      device.queue.submit([enc.finish()]);
      pickRead.mapAsync(GPUMapMode.READ).then(() => {
        const u32 = new Uint32Array(pickRead.getMappedRange().slice(0)); pickRead.unmap();
        let best = -1, bd = Infinity;
        for (let k = 0; k < PICK_W; k++) for (let x = 0; x < PICK_W; x++) {
          const occ = u32[(k * PICK_W + x) * FX + FS];
          const d = (x0 + x + 0.5 - px) ** 2 + (k - PICK_R) ** 2;
          if (occ && d < bd) { bd = d; best = occ - 1; }
        }
        if (best >= 0) { insp.slot = best; insp.pos = [px, py]; insp.last = 0; $('inspector').hidden = false; }
        insp.busy = false;
      }).catch(() => { insp.busy = false; });
      return;
    }
    // inspector: read the selected particle's genome, state and the field cells it senses
    if (insp.slot >= 0 && !insp.busy && now - insp.last > 250) {
      insp.busy = true; insp.last = now;
      device.queue.writeBuffer(reqs, 0, new Uint32Array([insp.slot, 0]));
      writeUniforms({ count: 1 });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, cGroups[cur]); pass.setPipeline(K.gather); pass.dispatchWorkgroups(Math.ceil(GS / 128));
      pass.end();
      enc.copyBufferToBuffer(freeList, 0, probeRead, 0, PROBE_G);
      enc.copyBufferToBuffer(parts, insp.slot * PBYTES, probeRead, PROBE_P, PBYTES);
      const x = Math.floor(insp.pos[0]), y = Math.floor(insp.pos[1]), r = insp.r;
      const cellAt = (cx, cy) => (((cy % G) + G) % G) * G + (((cx % G) + G) % G);
      [cellAt(x, y), cellAt(x + r, y), cellAt(x - r, y), cellAt(x, y + r), cellAt(x, y - r)].forEach((c, k) =>
        enc.copyBufferToBuffer(fieldBufs[cur], c * FX * 4, probeRead, PROBE_F + k * FX * 4, FX * 4));
      device.queue.submit([enc.finish()]);
      probeRead.mapAsync(GPUMapMode.READ).then(() => {
        const buf8 = probeRead.getMappedRange().slice(0); probeRead.unmap();
        const g = new Uint32Array(buf8, 0, GS).slice();
        const pf = new Float32Array(buf8, PROBE_P, PBYTES / 4), pu = new Uint32Array(buf8, PROBE_P, PBYTES / 4);
        const cells = Array.from({ length: 5 }, (_, k) => new Float32Array(buf8, PROBE_F + k * FX * 4, FX));
        insp.data = { g, pos: [pf[0], pf[1]], vel: [pf[2], pf[3]], energy: pf[4], age: pf[5], alive: pu[6], kids: pu[7], mem: [pf[8], pf[9]], cells };
        insp.pos = [pf[0], pf[1]];
        insp.r = Math.min(10, Math.max(1, Math.round(asF32(g[1]))));
        renderInspector();
        insp.busy = false;
      }).catch(() => { insp.busy = false; });
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
          const g = all.slice(j * GS, (j + 1) * GS);
          bank.add(g, { kids, age });
          considerOwn(g, kids + age / 2000, kids >= 2);
        }
        if (selected && !bank.entries.includes(selected)) selected = bank.entries.find(e => e.id === selected.id) || null;
        gatherPending = false;
        drawBank();
      }).catch(() => { gatherPending = false; });
    }
  }

  function frame(now) {
    rateFrames++;
    // GPU commitment: simulate on that share of frames, so the page uses roughly that share of the card
    commitAcc += ui.commit / 100;
    const simThisFrame = commitAcc >= 1;
    if (simThisFrame) commitAcc -= 1;
    const steps = paused || !simThisFrame ? 0 : ui.spf;
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
      if (!document.hidden) { workPending += steps * stats.alive; lifeWork += steps * stats.alive; }
      // keep your strain in play: when its share falls below 5%, re-plant its most advanced genome at your node
      if (stats.alive > 0 && (stats.mine < stats.alive * 0.05 || championChanges > plantedChampion) && now - lastStrainPlant > 20000) {
        plantedChampion = championChanges;
        lastStrainPlant = now;
        queuePlant({ genome: strainBest.g }, myNode.x * G, myNode.y * G, 16, 450);
      }
    }
    if (SHARED && now - lastSync > 60000) { lastSync = now; syncShared(); }
    if (SHARED && now - lastUpload > 45000) { lastUpload = now; uploadShared(); }

    runPrePass(now);

    // main uniforms: plant stage 1 clears its area via the eraser path this frame
    const p1 = !paused && plantQ.find(p => p.stage === 1);
    if (p1) p1.stage = 2;
    const seeding = seedProb > 0 && !paused;
    if (seeding) device.queue.writeBuffer(reqs, 0, strainBest.g);
    const sf = seeding ? seedStrainFrac : 0;
    if (p1) writeUniforms({ mouseOn: 1, mx: p1.x, my: p1.y, mr: p1.r, strainFrac: sf });
    else writeUniforms({ mouseOn: mouse.down && !paused ? 1 : 0, mx: ((mouse.x % G) + G) % G, my: ((mouse.y % G) + G) % G, mr: brushWorld(), strainFrac: sf });

    const enc = device.createCommandEncoder();
    const timeIt = hasTS && !tsPending && steps > 0;
    const pass = enc.beginComputePass(timeIt ? { timestampWrites: { querySet: tsSet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
    if (seedProb > 0 && !paused) {
      pass.setBindGroup(0, cGroups[cur]);
      pass.setPipeline(K.reset); pass.dispatchWorkgroups(1);
      pass.setPipeline(K.seed); pass.dispatchWorkgroups(wg);
      seedProb = 0;
      seedStrainFrac = 0.4;
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
    if (timeIt) { enc.resolveQuerySet(tsSet, 0, 2, tsResolve, 0); enc.copyBufferToBuffer(tsResolve, 0, tsRead, 0, 16); }
    stepsDone += steps; rateSteps += steps;

    cam.cx = ((cam.cx % G) + G) % G; cam.cy = ((cam.cy % G) + G) % G;
    camF[0] = cam.cx; camF[1] = cam.cy; camF[2] = cam.scale; camU[3] = mode;
    camF[4] = canvas.width; camF[5] = canvas.height; camF[6] = Math.max(0.9, cam.scale * 0.42); camF[7] = ui.fgain;
    camU[8] = strainTag; camU[9] = $('highlight').checked ? 1 : 0; camU[10] = $('pstyle').value === 'dots' ? 1 : 0;
    device.queue.writeBuffer(camBuf, 0, camData);

    const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    rp.setBindGroup(0, rGroups[cur]);
    rp.setPipeline(fieldPipe); rp.draw(3);
    rp.setPipeline(partPipe); rp.draw(6, N);
    rp.end();

    const readStats = !stats.pending && frameNo % 6 === 0;
    if (readStats) enc.copyBufferToBuffer(counters, 0, staging, 0, 192);
    device.queue.submit([enc.finish()]);
    if (timeIt) {
      tsPending = true;
      const nSteps = steps;
      tsRead.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(tsRead.getMappedRange().slice(0));
        tsRead.unmap();
        const ms = Number(t[1] - t[0]) / 1e6 / nSteps;
        if (ms > 0 && ms < 1000) simMs = simMs ? simMs * 0.9 + ms * 0.1 : ms;
        tsPending = false;
      }).catch(() => { tsPending = false; });
    }
    if (readStats) {
      stats.pending = true;
      staging.mapAsync(GPUMapMode.READ).then(() => {
        const c = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        stats.alive = c[2]; stats.births = c[3]; stats.deaths = c[4];
        stats.params = c[2] ? c[5] / c[2] : 0; stats.hidden = c[2] ? c[6] / c[2] : 0; stats.gen = c[7];
        stats.effects = Array.from(c.slice(16, 16 + NT));
        stats.mine = c[10];
        stats.uptakes = c[11]; stats.injections = c[12]; stats.bonded = c[13]; stats.learners = c[14]; stats.shared = c[15] / 100000;
        stats.pending = false;
      });
    }

    if (now - rateT > 500) {
      stepRate = rateSteps * 1000 / (now - rateT); fps = rateFrames * 1000 / (now - rateT);
      rateT = now; rateSteps = 0; rateFrames = 0;
      $('stats').textContent =
        `slots      ${N.toLocaleString()}\n` +
        `gpu sim    ${hasTS ? (simMs ? simMs.toFixed(2) + ' ms/step' : '…') : 'n/a (no timestamp-query)'}\n` +
        `field      ${G}×${G}×${F} (+density)\n` +
        `net        ${NI} in → ≤${HMAX} hidden → ${NO} out\n` +
        `avg params ${stats.params.toFixed(0)}  (max ${P})\n` +
        `avg hidden ${stats.hidden.toFixed(2)}\n` +
        `max gen    ${stats.gen}\n` +
        `births     ${stats.births.toLocaleString()}\n` +
        `deaths     ${stats.deaths.toLocaleString()}\n` +
        `weights    ${(stats.alive * stats.params / 1e6).toFixed(1)}M live\n` +
        `learners   ${stats.alive ? (100 * stats.learners / stats.alive).toFixed(0) : 0}% learn within life\n` +
        `bonded     ${stats.bonded.toLocaleString()} this step · ${stats.shared.toFixed(2)} E shared\n` +
        `genes      ${stats.uptakes.toLocaleString()} taken · ${stats.injections.toLocaleString()} injected`;
      renderStrains();
      const tot = Math.max(1, stats.alive * KE);
      $('effects').textContent = EFFECTS.map((n, t) => [n, stats.effects[t] / tot]).filter(([n]) => n !== 'none')
        .sort((a, b) => b[1] - a[1])
        .map(([n, f]) => `${n.padEnd(9)} ${'█'.repeat(Math.round(f * 60)).padEnd(14)} ${(f * 100).toFixed(1)}%`).join('\n');
    }
    placeRing();
    placeNodes();
    if (frameNo % 15 === 0) drawWorld();
    if (!fitDone && !paused && now - bootT > 3000) {
      if (now - bootT < 9000) fitSamples.push(fps);
      else {
        fitDone = true;
        const avg = fitSamples.reduce((a, b) => a + b, 0) / Math.max(1, fitSamples.length);
        let tries = 0; try { tries = +(sessionStorage.getItem('fieldlife-fittries') || 0); } catch {}
        if (avg < 22 && N > 16384 && tries < 3) {
          const next = Math.max(16384, Math.floor(N / 2 / 128) * 128);
          try { localStorage.setItem('fieldlife-n', next); sessionStorage.setItem('fieldlife-downsized', next); sessionStorage.setItem('fieldlife-fittries', tries + 1); } catch {}
          saveBank(); location.reload(); return;
        }
        try { localStorage.setItem('fieldlife-n', avg > 50 && N < N_GUESS ? Math.min(N_GUESS, N * 2) : N); } catch {}
      }
    }
    if (fitNote && now - bootT > 12000) fitNote = '';
    $('hud').textContent = `zoom=${(cam.scale / cam.fit).toFixed(1)}x  pop=${stats.alive}  ${stepRate.toFixed(1)} steps/s  ${fps.toFixed(0)} fps  ${canvas.width}px  bank=${bank.entries.length}${paused ? '  [paused]' : ''}${fitNote ? '\n' + fitNote : ''}`;
    frameNo++;
    requestAnimationFrame(frame);
  }

  // Debug helpers (used by headless checks).
  async function fieldSummary() {
    const size = G * G * FX * 4;
    const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(fieldBufs[cur], 0, rb, 0, size);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(rb.getMappedRange());
    const mean = new Array(FS).fill(0), max = new Array(FS).fill(0);
    for (let i = 0; i < f.length; i++) { const c = i % FX; if (c >= FS) continue; mean[c] += f[i]; if (f[i] > max[c]) max[c] = f[i]; }
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
