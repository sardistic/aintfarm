# Decisions

## 2026-11-07 — Separate visual signal field from agent behavior

The four-channel diffusing color field is a Canvas display effect inspired by the supplied Shadertoy example. Agent decisions continue to use the existing food and home fields, and sand continues to use the granular solver. This keeps the visual change from silently changing the evolutionary experiment.

## 2026-11-07 — Log significant events and periodic snapshots

The browser keeps the latest 1,500 successful dig, drop, harvest, share, birth, and death events, plus snapshots every 10 simulated seconds. The current run can be exported as JSON. The log is in memory and resets with the world or browser reload.

## 2026-11-07 — Connect visible signals to behavior and make the colony ecological

The earlier four-channel field was display-only. It now holds concentrations emitted by agents with inherited, mutable signatures. Different channels diffuse and decay at different rates; local gradients weight movement probabilities while the neural controller still supplies action probabilities. Surface versus burrow foraging preference is inherited through the first signature component and constrains surface-preferring agents from falling into the shaft. This preserves both food regions as active niches.

Successful actions and attempts are counted separately. Workers carry harvested food to a finite nest store, draw from it near connected nest tunnels when weak, dig toward buried food, and move sand toward spoil areas. Reproduction requires prior harvests, sufficient energy, a cooldown, and room under a resource-linked population ceiling. Founders start at varied ages, with an age limit of 540 simulated seconds, so generations turn over instead of all dying simultaneously. This is a designed selection environment with innate drives, not within-life training or a reconstruction of ant neurology.

The packed terrain and tunnel contour are cached and recomputed on packed-grain changes; loose grains remain drawn dynamically. This keeps the visual field tied to simulation state while avoiding full terrain work on every grain frame.

## 2026-11-07 — Make mutations, chemistry, and ancestry observable

Each offspring now inherits and mutates a phenotype with 12 physical traits, four emission channels, four independent receptor channels, and pigment in addition to its 9,904 neural weights. Mutability controls the neural mutation rate, and occasional 96-weight block changes and large trait/channel jumps create divergent branches. A lineage means all descendants of one starting founder; generation counts births along that parent chain. The full family tree and mutation details are retained for JSON export, while the live UI shows dominant founder families, ancestry, trait bars, parent-to-child deltas, and highlights living relatives.

Chemical concentrations affect movement through receptor gradients, and neighboring agents also react directly through their receptor/emission compatibility. Compatible encounters can transfer energy and produce nutrient blooms; incompatible encounters can drain energy. Carcasses also return nutrients to the environment. These reactions are intentionally designed rules, not learned within-life behavior. The manual food-placement tool was removed so food availability comes from the simulation's resources, blooms, and deaths.

The farm canvas now renders at a device-aware resolution up to 3 times the simulation dimensions. Packed grains are jittered within their cells and drawn as discrete high-contrast marks; tunnel boundaries and agents are drawn with thin, sharp strokes. Blur and shadow effects were removed from the farm image, while the chemical field remains a subdued background layer. Food visibility, terrain physics, and browser interactions remain in world coordinates.

Food classifications are cached between field updates, and the home-distance field is rebuilt only when packed topology changes. This avoids rescanning all resources for every agent and rebuilding routes when only loose grains move.

## 2026-11-07 — Keep zoom as a view transform

The farm camera applies the same CSS translation and scale to both world canvases. It zooms toward the cursor, clamps panning to the farm bounds, and leaves simulation coordinates and exported images unchanged. The existing overlay-to-world pointer mapping uses the transformed canvas bounds, so sand tools and agent inspection still target the visible location. Mouse wheel, zoom buttons, Space or middle-button drag, and two-finger gestures control the camera. This keeps camera movement independent of physics and agent behavior.

## 2026-11-07 — Render the culture with a GPU reaction field

The packed terrain, nutrient concentration, and four simulation chemical channels are uploaded to a WebGL renderer. A fragment shader draws individual mineral grains, water caustics, nutrient colonies, and chemical colors. Two GPU textures alternate through a Gray-Scott-style reaction-diffusion step seeded at nutrient patches. This reaction field is visual and does not alter agent behavior or physical resources; the input nutrient and chemical channels do reflect live simulation state. The existing 2D canvas renders loose grains and creatures above the GPU scene, and remains the fallback when WebGL is unavailable. Both layers share the camera transform and image export. The GPU path avoids the previous full packed-grain CPU raster on initial load. The CPU fallback also avoids thousands of per-cell fill calls.

## 2026-11-07 — Give the culture a dense motile population

The reference video has roughly 400,000 moving colored particles that form and break apart into structures. Static nutrient glow and a visual reaction shader did not create that motion. The farm now runs 36,000 smaller cells alongside the neural ant-like agents. Cells use compact inherited traits (four chemical receptors, emission channel, speed, and turning tendency), move with a circulating flow and local chemical gradients, feed near the same nutrient patches at a low rate, and die or reproduce with mutation. They deposit into a four-channel local field and weakly into the ant-like agents' signal field, coupling the two populations. The cells are CPU simulated and GPU rendered as colored beads with short trails, not 9,904-parameter neural agents. Canvas fallback uses 12,000 cells. The population is fixed in size but has births and deaths through slot replacement; its counters and genome sample are included in run-log export. Culture stepping is capped to approximately 30 updates per browser second to leave room for the neural simulation and rendering.

A 20-second snapshot of the first motile implementation showed cells collapsing into food-source points despite traveling quickly. The force now targets an orbit radius with repulsion near the center, nearby velocity alignment, and high-density self-repulsion in the chemical response. Feeding reaches beyond the center and energy drains more slowly, preserving broad moving populations between patches. Short lagged position trails make the actual particle paths visible.

Two additional GPU textures alternate as a fading colored trail field. Each frame blends the previous trail with slight diffusion and stamps the current moving cells, then the pond shader composites those wakes behind the bright cells. The trail field is reset with a new world and freezes when the simulation is paused. This reveals collective motion without adding extra CPU drawing per cell.

## 2026-11-08 — Add a legible microorganism ecology

The motile particle culture was visually active but lacked organisms with recognizable needs and consequences. A new population of 84 membrane-bound organisms swims over the black sand substrate, with up to 170 through division. Each has inherited mutable motility, aggression, adhesion, growth, metabolism, mutation rate, hue, morphology, and chemical preference. Their state machine covers seeking, gathering, hunting, fighting, fleeing, resting, division, and death. A spatial bucket index in the 36,000-cell culture lets organisms capture nearby cells at bounded cost. Capture transfers energy and mass to the organism, causing visible growth, while the captured cell is replaced elsewhere by a mutated descendant. Organisms attract nearby motile cells through a coarse force grid and emit mutable chemical signatures into the field sensed by neural agents. Predation transfers energy and damages rivals from other families. Division shares body mass and energy with a mutated child; death releases motile cells. These interactions run in simulation state; the membrane and event effects render that state, and the inspector, activity strip, event log, family board, and JSON export expose it. Five-second snapshots summarize the ecology over time. The ant-like neural population and sand physics remain present below this liquid ecology.

The membrane radius follows the square root of accumulated mass. Feeding can grow a mature organism to roughly 32 world pixels in radius, while larger bodies swim more slowly. The population cap allows some mature organisms to keep growing after division fills the dish. A 60-second, 12,000-small-cell validation run had 169 living organisms, visible growth to the radius cap, continuing divisions and deaths, and no extinction.

## 2026-11-08 — Field Life: separate WebGPU branch following the reference mechanism

The ant farm accumulated layers that never matched the reference: a CPU ant brain, granular sand, rule-based cells, and membrane organisms. `public/field/` is a clean rebuild with no shared code. It follows the author's description: particles at continuous positions over a G×G×F field tensor, with no direct particle–particle interaction. Each particle has its own untrained, mutating MLP and a mutable chemistry rule. Energy is injected by shifting Gaussian sources, and particles split into mutated children.

All state lives in WebGPU storage buffers. Each step runs five compute dispatches: reset counters, particle step, collect free slots and stats, spawn children into free slots, and field diffusion/decay/injection. Field writes use `atomic<i32>` deposits in 2^-18 fixed point because WebGPU has no float atomics. Births never race with deaths because the free list is rebuilt after the particle step and before spawning. Genomes are interleaved (word k of slot s at k·N + s) so that neighboring threads read coalesced memory. In headless Chrome on the user's Pascal GPU, the strided layout ran at about 6 steps/s and the interleaved one at about 60, at similar population.

Two balance rules were needed so that the field carries information. First, writing information costs 1/10 the energy of writing energy, and absorbing information yields 1/10 the energy. Second, metabolic burn is written as exhaust along the net's aim. Without these, selection removed writing entirely, and the info channels measured about 0.005 at peak.

## 2026-11-08 — Mutable physics: effect genes, density channel, gene bank

Particles can now mutate into qualitatively different physics. Each genome has 4 effect slots of (type | channel a | channel b, p0, p1, p2). Mutation can retype a slot to any of 13 primitives, rewire its channels, and retune its parameters. The net gates each slot on every step, and each expressed effect has an energy upkeep. Every primitive acts through the field or the particle's own velocity, so particles still never interact directly. Effects that move value between channels conserve energy, given that a unit of info is worth 1/10 of a unit of energy. In the effect switch, values are re-read from the field buffer instead of dynamically indexing private arrays, because that indexing spilled registers and cut throughput by about 3×.

The field gained a non-evolvable density channel (occupancy deposited with atomics, lightly smoothed). Nets sense it. Baseline physics applies pressure from the range-1 density gradient and cohesion along the gradient at the evolved sensing range. Together these act as surface tension, intended to favor droplets and boundaries over a uniform spread.

The gene bank is a CPU-side nearest-neighbor archive (unstructured MAP-Elites/CVT-style), not a database service. Genome transfer reuses existing buffers so that no compute storage bindings are added beyond the default limit of 8:
- **Upload.** `reqs` doubles as the upload area. Planting is a two-frame operation. In the first frame, the eraser path frees the area. In the second, a `plant` kernel fills slots from the previous frame's free list.
- **Download.** `freeList` doubles as the download area for `gather`.

Uploads and downloads run in their own submits before the frame's steps, so they never race the free list or split requests. Only genomes with at least one offspring enter the bank. The bank persists in localStorage, keyed by F and HMAX.

## 2026-11-08 — Publish as aintfarm.sardistic.com

Field Life is the product, so it serves at `/`. The ant farm stays reachable at `/farm`, unchanged, because its absolute asset paths still resolve from there. Production is a single non-root Node container built from this repository's `main` branch, matching the house pattern for small public apps: a Git checkout on the host, Docker Compose, Cloudflare Tunnel, and no published ports. Host-specific details live in the private infrastructure runbook, not here.

The server sends `Cache-Control: no-cache` so deploys are visible at once through Cloudflare. It answers HEAD requests, for probes, and exposes `/healthz` for the container healthcheck. Pages carry the sardistic eye favicon, the shared `sardize/transit.js` cross-site transition, and a fixed eye link back to sardistic.com.

## 2026-11-08 — Gene transfer, kin bonding, within-life learning

The field gained a second hidden slot per cell (stride F+2). It holds the id of a particle that occupied the cell in the previous step, written with `atomicMax` and carried across the field step as bits in an f32. This is the only way particles can find a neighbour. It keeps the field as the medium and avoids adding a ninth storage binding.

Three effect types use it: uptake (copy a gene in), inject (copy a gene out) and bond (spring plus velocity matching toward kin, judged by bond-key Hamming distance). Gene copies race benignly with the donor's own updates. Learning updates W2 in place in the genome: two packed f16 weights per word, every other step, only when the learning-rate gene is above 5e-4. That makes inheritance Lamarckian, and it avoids keeping a second weight copy, which would double genome memory.

In a 60 s run at 131k slots, inject rose from 4.1% to 7.4% of gene slots, bond from 3.2% to 5.0%, and learners from 53% to 58%. The genome is now 371 words, and old gene banks are keyed separately (v3).
