# aintfarm

Live at <https://aintfarm.sardistic.com>. **Field Life** (the WebGPU neural-particle world, `public/field/`) is the homepage. The original ant farm below is at `/farm`.

## Art Farm: Microbial World

A local browser simulation of a microbial world over a black sand substrate. Larger membrane-bound organisms swim through the culture, collect smaller cells, grow, hunt rival families, rest, divide into mutated offspring, and die back into the cell population. The original ant-like agents still work the sand with distinct, untrained 9,904-parameter recurrent controllers. A second population of 36,000 motile cells makes the culture move at a dense scale.

WebGL draws individual mineral grains, refracted water, nutrients, emitted chemistry, motile cells, and fading motion wakes. A GPU reaction-diffusion film grows around nutrient patches. Small cells have mutable receptors, chemical type, speed, and turning tendency. Membrane-bound organisms have inherited motility, aggression, adhesion, growth, metabolism, mutation rate, color, morphology, and chemical preference. They emit their own chemistry into the field sensed by neural agents. Collecting nearby cells transfers energy and mass; predation damages a rival; division splits resources between parent and child; death releases cells into the culture. The organisms swim over the sand substrate while ant-like agents move sand within it. Without WebGL, a smaller motile population renders on Canvas.

This is an artificial life experiment with ant-like surroundings, not a reconstructed ant brain. The 36,000 cells use compact rules, not 9,904-parameter neural controllers. It is not yet a distributed GPU cellular automaton or a 400,000-agent run like the reference video.

## Run

Requires Node.js 20 or newer:

```sh
npm start
```

Open <http://localhost:3000>. The farm fills the browser viewport inside a narrow frame. The left tool rail moves an individual grain, removes grains, pours loose grains, or inspects life. Choose the inspect tool and click a bright membrane to see what that organism is doing, its size, energy, family, and inherited traits. The activity strip shows seeking, gathering, hunting or fighting, and resting counts. The controls panel shows captures, attacks, divisions, deaths, evolution events, run-log export, image export, and Jev. The starting neural-agent slider supports 30–800; microbial organisms start at 84 and can divide up to a population limit of 170.

Use the mouse wheel over the farm to zoom toward the cursor, or use the − and + buttons (100% resets the view). Hold Space and drag, or drag with the middle mouse button, to pan. On touch screens, use two fingers to pinch and pan. Zoom ranges from 100% to 800%; grain tools and agent inspection follow the zoomed view.

Run `npm test` for simulation checks.

## Rules

- A 32-input, 64-unit, 96-unit, 16-output controller has exactly 9,904 inherited weights and biases per creature. There is no backpropagation or per-action API call. A stochastic draw from its output distribution selects an action.
- Available actions are four local moves, four adjacent digs, dropping sand, harvesting nearby exposed food, sharing energy, emitting food or home signals, reproduction, rest, and exploration. The neural output is weighted by viability and innate food, home, excavation, and colony-signal drives. Attempted and successful actions are counted separately.
- Food patches replenish at limited rates and some start buried. Agents dig toward buried sources, harvest food, and bring units to a finite nest reserve. Dead agents can become carcass nutrients. Compatible chemical encounters can produce new nutrient blooms; incompatible encounters can drain energy. The resource ecology runs without food placement by the user.
- A lineage is the family of every descendant of one starting agent. Generation counts births from that founder. Offspring mutate neural weights, sometimes a 96-weight block, plus 12 physical traits, four emission channels, four independent receptor channels, and pigment. Trait mutations affect movement, digging, foraging niche, social transfer, aggression, secretion, sensing range, size, metabolism, fecundity, longevity, and mutation rate. The inspector shows parent-to-child changes, while the family board shows which lineages are surviving.
- Signals diffuse and react at different rates. Emissions and receptors can mutate separately, changing attraction, repulsion, cooperation, conflict, and nutrient blooms. The same concentrations drive the subdued field image. There is no within-life training or explicit fitness score: energy and offspring are the selection pressure. A population ceiling, staggered lifespans, and a 240-unit nest store support turnover without unbounded growth. The run log retains 1,500 recent events and 10-second snapshots in browser memory; export JSON before resetting.
- The small cells move through a four-channel local chemical field, align with nearby motion, circulate around shared nutrient patches, and avoid crowding at their centers. They feed from those patches at a low rate and emit weak signals into the larger agents' field. Starvation and age replace cells with mutated descendants of viable neighbors. The controls show cell births and large mutations; JSON export includes chemistry counts and a sample of current genomes.
- Membrane-bound organisms draw small cells toward themselves and absorb them from a spatial particle index. Feeding increases energy and body mass, making the membrane visibly larger. Aggressive organisms chase smaller rivals from other families; contact transfers energy and causes wounds or death. Organisms can seek food, gather, hunt, fight, flee, rest, divide, and die. Division inherits traits with occasional large changes in pigment, morphology, chemistry, and behavior. Dead bodies release motile cells. The activity strip and event log report these outcomes, and JSON export includes the microbial family tree and recent events.
- The world has roughly 118,000 separate packed grains at startup. Each is about 1.4–1.9 world pixels across, while an agent body is only a few pixels long. Packed grains use a 2-pixel lattice for terrain but are jittered and rendered into a high-resolution, unblurred canvas; loose grains retain free positions and velocities. Gravity, collision separation, weak cohesion, and damping make dense streams flow and settle. This is an approximate CPU 2D granular model, not the supplied GPU fluid solver.
- The old scripted colony engine remains in `public/simulation.js` for comparison and its regression tests. The page runs `public/evolution.js` with grain physics in `public/granular.js`.

## Field Life branch (`/field/`)

A separate WebGPU rebuild that follows the mechanism described for the reference video. It does not use any code from the ant farm above. Open <http://localhost:3000/field/>. It requires Chrome or Edge with WebGPU.

- **Field tensor.** The world is a toroidal 768×768×(8+1) grid. Channel 0 holds energy. Twelve drifting, pulsing Gaussian "suns" inject energy into it. Channels 1–7 hold information, and each diffuses and decays at its own rate. A ninth density channel records how many particles occupy each cell.
- **Particles never touch.** Up to 262,144 particle slots sit at continuous positions. Each step, a particle reads the field at its cell, absorbs from it, and writes back to it with fixed-point atomics. Baseline surface tension, short-range pressure against cohesion at the particle's sensing range, also goes through the density channel.
- **Chemistry.** Each particle has its own mutable rule: per-channel absorb rate, energy gained per unit absorbed (can be negative), and push along or against that channel's gradient. Info channels hold 1/10 energy per unit, so absorbing them can't create energy.
- **Effect genes.** Each particle has 4 slots, and mutation can retype any of them to one of 13 physics primitives. The particle's net gates each slot's strength on every step. Every slot has channel wiring (a, b) and parameters.
  - **Motion:** taxis, swirl, isoline-seeking (settles on a concentration contour, which makes membranes), jet, anchor, pressure/cohesion, heat, global drift.
  - **Chemistry and signalling:** catalysis a→b (value-conserving, keeping a cut), pulsed emission, siphon at range, signal relay a→b (excitable waves), toxin shield.
- **Its own net.** Every particle has a unique, untrained MLP. Its 33 inputs are field values and x/y gradients for all channels including density, plus energy, velocity, 2 self-written memory registers, and an internal clock with an evolved frequency. The hidden layer is 2–12 tanh units, and that width is a gene. The 18 outputs are split, the child's share, a write gate and amount, an 8-dim aim, 2 memory values, and 4 effect gates. Nets range from 72 to 642 parameters, stored as f16 in an interleaved, coalesced GPU layout.
- **Energy.** Every action costs energy. Metabolism scales with network size, each expressed effect has an upkeep, and the burned energy leaves as exhaust in the info channels along the net's aim. Particles die when their energy runs out or at an inherited maximum age.
- **Selection and mutation.** A split takes a free slot. The child gets mutated chemistry, effects (retype, rewire, retune), weights, sensing range, clock, mutation rate, lifespan, and sometimes ±1 hidden unit. Slots are the carrying capacity. The effect histogram in the panel shows which physics selection is favoring.
- **Gene bank (vector database of evolved nets).** Every second, 64 random slots are copied from the GPU. Genomes that have reproduced are embedded as a 49-dim vector of chemistry, effect loadings, architecture and range, and an 8-dim random-projection fingerprint of the weights.
  - **Niches.** A genome within the niche radius of an existing entry replaces it only if it's fitter (more offspring, plus age). Otherwise it becomes a new niche. When the bank is full (400), the weaker member of the closest pair is dropped and the radius grows.
  - **Map and planting.** The map is a live PCA of the bank. Click a dot to see that species' effects, diet, and net size, then plant it with the plant tool. *Immigration* plants a random niche every 15 s to counter monocultures, and archived species also reseed a collapsed dish.
  - **Persistence.** The bank persists in browser storage and can be exported or imported as JSON (`?fresh` ignores the stored bank).
- **Color.** Particle color comes from the chemistry and effect genes, tinted by an inherited lineage hue, so similar colors mean similar phenotype.
- **Controls.** Left-drag erases, or plants with the plant tool. Scroll zooms and right- or middle-drag pans. Space pauses. Keys 1–5 switch views (field + particles, particles, field, sunlight, density), and E and P switch tools. Sliders set sunlight, metabolism, parameter cost, mutation, pressure, cohesion, and steps per frame.
- **URL parameters.** `n` sets the slot count, `g` the grid size, `f` the channel count, `h` the maximum hidden width, `mode` the initial view, and slider ids (such as `spf=3`) set slider values. `noimm` disables immigration and `fresh` starts with an empty bank. The slot count is reduced automatically to fit the GPU's buffer limit.

**Differences from the reference.** Its particles average about 10k parameters across 400k nets on a 1k×1k×32 field, running across multiple GPUs. Here, the 262k nets have about 70–640 parameters each in one browser tab. The effect library, the density channel, surface tension, and the gene bank are this branch's own additions. The thread does not describe them.

## Jev

To enable **Ask Jev to look**, set `TYPESAFE_API_KEY` in the server process environment before starting. Keep the key out of browser code and committed files. Clicking the button sends a JPEG of the current world to TypeSafe's Jev image endpoint for visible chamber and branch scores. Jev's scores are observations and do not alter agent actions, reproductive fitness, or energy. No image is sent automatically. The simulation runs without Jev.
