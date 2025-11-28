speed = 0.35

veins = voronoi(18, 0.03, 2.2)
  .modulate(noise(1.8, 0.25), 0.35)
  .luma(0.72, 0.15)
  .color(0.10, 0.22, 0.28)
  .saturate(0.5)
  .brightness(-0.6)

filaments = osc(6.5, 0.02, 0.12)
  .modulate(noise(1.2, 0.35), 0.5)
  .luma(0.67, 0.12)
  .color(0.12, 0.26, 0.30)
  .brightness(-0.65)
  .rotate(() => Math.sin(time * 0.05) * 0.05)

glow = filaments
  .scale(1.01)
  .saturate(0.4)
  .brightness(-0.55)
  .colorama(0.02)

pattern = veins
  .add(glow, 0.6)
  .add(filaments, 0.35)
  .modulateScale(noise(0.6, 0.2), 0.015)
  .modulateRotate(noise(0.8, 0.15), 0.02)
  .contrast(1.1)
  .saturate(0.55)
  .hue(() => Math.sin(time * 0.03) * 0.02)

pattern
  // Soft circular containment so it stays sparse and centered
  .mask(shape(100, 0.9, 0.9))
  .out(o0)
