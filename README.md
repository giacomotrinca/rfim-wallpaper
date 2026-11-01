# RFIM Wallpaper

```
 ╔════════════════════════════════════════════════════════════════╗
 ║                                                                ║
 ║    R F I M   W A L L P A P E R                                 ║
 ║                                                                ║
 ║    avalanches · crackling · critical phenomena                 ║
 ║    disordered systems on the GPU                               ║
 ║                                                                ║
 ╚════════════════════════════════════════════════════════════════╝

    ↑ ↓ ↑ ↓ ↑ ↓ ↑ ↓ ↑ ↓ ↑ ↓          ↑  ↑  ↓  ↓  ↑  ↓
    ↓ ↑ ↓ ↑ ↓ ↑ ↓ ↑ ↓ ↑ ↓ ↑          spin configuration
    ↑ ↓ ↑ ↑ ↓ ↓ ↑ ↓ ↑ ↑ ↓ ↓                ▼
    ↓ ↑ ↓ ↓ ↑ ↑ ↓ ↑ ↓ ↓ ↑ ↑          a domain wall
```

Turn statistical mechanics into desktop art. `rfim-wallpaper` simulates a
**Random Field Ising Model** on the GPU (PyTorch/CUDA) and renders the resulting
domain walls as fractal "ember" textures — quenched disorder, Barkhausen
avalanches, crackling noise, and critical phenomena, at native resolution.

## What it does

Two dynamics, one Hamiltonian

```
H = -J Σ_<ij> s_i s_j - Σ_i h_i s_i - H_ext Σ_i s_i ,   h_i ~ N(0, Δ)
```

| mode | physics | knobs |
|------|---------|-------|
| `avalanche` | **T = 0** field ramp → Barkhausen crackling. Spins start all-down and `H_ext` is ramped; each step relaxes to the unique ground state (Middleton *no-passing* theorem). The bursts of flips are the avalanches. | `--disorder`, `--ramp-steps`, `--target-m` |
| `thermal` | **finite T** Metropolis Monte Carlo, vectorized on the checkerboard. `Δ=0`, `T=T_c≈2.269` reproduces the clean 2D Ising critical point; `Δ>0` pins and roughens the walls (Imry–Ma disorder in `d=2`). | `--temperature`, `--disorder`, `--field`, `--sweeps` |

One spin = one pixel, so there is **no upscaling blur** — run it at native
resolution up to 4K.

## Install

```bash
conda create -n wallpaper python=3.12 -y
conda activate wallpaper
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or the CUDA build for your driver
pip install -e .
```

Verify the GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # True
```

## Quickstart

```bash
rfim-wallpaper --resolution fhd                          # default: embers, native-like, T=1.6
rfim-wallpaper --preset pulito --resolution qhd          # clean Ising at T_c, 2K
rfim-wallpaper --preset crackle                          # T=0 avalanches
rfim-wallpaper --colors matrix                           # solid theme
rfim-wallpaper --colors plasma,lime --mix clouds         # two-color zones
```

Output: `~/rfim-wallpaper.png` (override with `--out`).

## Gallery

| Embers · thermal | Avalanche · matrix |
|---|---|
| ![braci-thermal](examples/braci-thermal.png) | ![matrix-avalanche](examples/matrix-avalanche.png) |

| Two-color blend (`clouds`) | Hard zones (`zones`) |
|---|---|
| ![plasma-lime](examples/plasma-lime-clouds.png) | ![neon-cobalt](examples/neon-cobalt-zones.png) |

## Color themes

14 themes ship in [`colors.toml`](rfim_wallpaper/colors.toml) — edit it to add
your own.

```
braci · plasma · matrix · neon · synthwave · abyss · blue
lava · aurora · acid · lime · cobalt · solar · graphite
```

```bash
rfim-wallpaper --list-colors          # print them with swatches
rfim-wallpaper --colors neon          # a single theme
rfim-wallpaper --colors blue,lime     # blend zones of two themes
```

`--mix` controls how multiple colors combine:

| mode | look |
|------|------|
| `clouds` (default) | smooth organic zones |
| `gradient` | diagonal blend |
| `radial` | center vs. edges |
| `angular` | around the center |
| `zones` | hard Voronoi-like cells |

## Simulation presets

Six named recipes in [`presets.toml`](rfim_wallpaper/presets.toml), each targeting
a distinct regime:

| preset | mode | character |
|--------|------|-----------|
| `pulito`  | thermal   | clean Ising at T_c — critical fractal |
| `fumo`    | thermal   | below T_c — large ordered domains (sparse) |
| `torbido` | thermal   | above T_c — fine paramagnetic grain (dense) |
| `braci`   | avalanche | coercive field — balanced crackling |
| `isole`   | avalanche | asymmetric — minority "islands" |
| `caos`    | avalanche | strong disorder — fragmented chaos |

```bash
rfim-wallpaper --list-presets
rfim-wallpaper --preset fumo          # default: sparse
rfim-wallpaper --preset caos          # fragmented
```

Any flag overrides the preset: `--preset fumo --disorder 0.5 --sweeps 1000`.

## Reproducibility

Same seed → same image, even on GPU: the quenched field and grain are generated
deterministically on the CPU, and the GPU only runs deterministic math.

```bash
rfim-wallpaper --seed 42 --resolution fhd --out a.png
rfim-wallpaper --seed 42 --resolution fhd --out b.png
cmp a.png b.png                        # identical
```

## CLI reference

```
--resolution {hd,fhd,qhd,uhd,native}   output resolution (1 spin = 1 px)
--width / --height                     free-form size
--style {texture,material}             rendering style
--mode {thermal,avalanche}             dynamics
--preset NAME                          parameter recipe
--colors A[,B,...]                     color theme(s)
--mix {clouds,gradient,radial,angular,zones}
--disorder Δ · --temperature T · --field H
--sweeps N · --ramp-steps N · --target-m M
--seed N · --cpu · --out PATH
--list-colors · --list-presets
```

## Why it's fast

The whole pipeline runs as GPU tensors: quenched-field generation, the
nearest-neighbour sums (`torch.roll`), the zero-T front relaxation and the
Metropolis sweeps, the domain-wall extraction, and the Gaussian blur. A
1920×1080 simulation thermalizes in a few seconds on an RTX 3060.

## License

[MIT](LICENSE)
