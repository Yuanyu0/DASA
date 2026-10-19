# DASA project page

Open `index.html` to preview the static project page. The reading flow is observation → method → results. Each of the paper’s 11 figures has its own explanatory slide. Full-size images in `assets/figure-01.webp` through `assets/figure-11.webp` are high-resolution WebP exports rendered directly from the matching PDFs in `../nips 2026 figures/` at 2.5× scale. The page uses smaller files in `assets/thumbs/` and loads the full image only when a figure is opened. The three video slots use compact posters in `assets/posters/`. The three additional PDF exports are in `assets/pdf-extras/`.

## Add demo videos

The video showcase has one slot each for 480P, 720P, and 1080P. Each tile has a `data-slot` name matching its resolution.

Add an MP4 under `videos/`, then replace the contents of the matching `.video-slot` with:

```html
<video controls playsinline preload="metadata" poster="assets/posters/480p.webp">
  <source src="videos/baseline-480p.mp4" type="video/mp4">
</video>
```

Use the poster that matches the resolution: `posters/480p.webp`, `posters/720p.webp`, or `posters/1080p.webp`. Keep the surrounding `.video-slot` element so the three-card row stays aligned.
