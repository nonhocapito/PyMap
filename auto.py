#!/usr/bin/env python3
"""
topo_map.py — Generatore di mappe topografiche in scala con stile OpenTopoMap.

Uso:
    python topo_map.py --lat 45.8 --lon 10.6
    python topo_map.py --lat 45.8 --lon 10.6 --scale 50000 --format A3 --dpi 200
    python topo_map.py --lat 40.167 --lon 18.017 --scale 13449 --format A3 --landscape
    python topo_map.py --lat 41.9028 --lon 12.4964 --out roma_centro.png --no-grid

Dipendenze:
    pip install requests Pillow
"""

import csv
import math
import time
import argparse
import io
from pathlib import Path

try:
    import requests
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("Installa le dipendenze:\n  pip install requests Pillow")

try:
    from pyproj import Proj, Transformer
except ImportError:
    raise SystemExit("Installa pyproj:\n  pip install pyproj")

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

TILE_URL      = "https://tile.opentopomap.org/{z}/{x}/{y}.png"
TILE_SIZE     = 256
EARTH_CIRC    = 2 * math.pi * 6_378_137
REQUEST_DELAY = 0.2           # secondi tra tile (rispetta rate limit del server)
#MAX_TILES     = 200
MAX_TILES     = 800

HEADERS = {
    "User-Agent": "TopoMapGenerator/1.0 (https://github.com/example/topomap)"
}

PAPER_SIZES = {          # larghezza x altezza in mm, orientamento ritratto
    "A4":  (210, 297),
    "A3":  (297, 420),
    "A2":  (420, 594),
    "A1":  (594, 841),
    "A0":  (841, 1189),
}

# Colori griglia alfanumerica (stile get-map.org)
GRID_LINE_COLOR   = (100, 100, 180, 160)   # blu-grigio semitrasparente
GRID_LABEL_BG     = (255, 255, 255, 200)   # sfondo etichette
GRID_LABEL_COLOR  = (40, 40, 40)           # testo etichette

# Colori griglia UTM
UTM_LINE_COLOR    = (200, 50, 50, 180)     # rosso semitrasparente
UTM_LABEL_BG      = (255, 240, 240, 220)   # sfondo etichette UTM
UTM_LABEL_COLOR   = (160, 0, 0)            # testo etichette UTM


# ---------------------------------------------------------------------------
# Cartografia: conversioni
# ---------------------------------------------------------------------------

def deg2global_pixel(lat: float, lon: float, zoom: int) -> tuple:
    """Coordinate geografiche -> pixel globali (Web Mercator)."""
    lat_r = math.radians(lat)
    n = 2 ** zoom
    px = (lon + 180) / 360 * n * TILE_SIZE
    py = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n * TILE_SIZE
    return px, py

def meters_per_pixel(lat: float, zoom: int) -> float:
    """Risoluzione in metri/pixel a una data latitudine e zoom."""
    return EARTH_CIRC * math.cos(math.radians(lat)) / (2 ** zoom * TILE_SIZE)


def best_zoom_and_resize(lat: float, scale: int, dpi: int) -> tuple:
    """
    Calcola il livello di zoom ottimale e il fattore di resize da applicare
    dopo il download per ottenere ESATTAMENTE la scala richiesta.

    Strategia: sceglie sempre lo zoom piu' dettagliato (z_high = ceil),
    poi riduce (downscale) per raggiungere la scala target. Il downscale
    preserva la nitidezza meglio dell'upscale.

    Ritorna (zoom: int, resize_factor: float)
        resize_factor < 1  -> riduzione (tipico)
        resize_factor = 1  -> nessun resize necessario
        resize_factor > 1  -> ingrandimento (zoom cappato a MAX_ZOOM=17)

    Nota: OpenTopoMap serve tile solo fino a zoom 17. Superato quel livello
    il server risponde 400 Bad Request. Il cap garantisce che lo zoom non
    superi mai 17; se la scala richiede piu' dettaglio si ricorre a un
    lieve upscale (resize_factor > 1).
    """
    MAX_ZOOM = 17   # limite massimo tile server OpenTopoMap
    needed_mpp = scale * (0.0254 / dpi)
    z_float = math.log2(
        EARTH_CIRC * math.cos(math.radians(lat)) / (needed_mpp * TILE_SIZE)
    )
    # Preferisci sempre il ceil (piu' dettaglio, poi downscale),
    # ma non superare MAX_ZOOM per evitare HTTP 400.
    zoom = min(math.ceil(z_float), MAX_ZOOM)
    # Fattore di resize: mpp_scaricato / mpp_necessario
    # > 1 quando lo zoom e' stato cappato (upscale lieve, inevitabile)
    mpp_at_zoom = meters_per_pixel(lat, zoom)
    resize_factor = mpp_at_zoom / needed_mpp
    return zoom, resize_factor

def effective_scale(lat: float, zoom: int, dpi: int) -> int:
    """Scala effettiva risultante da zoom + dpi (senza resize)."""
    return int(round(meters_per_pixel(lat, zoom) / (0.0254 / dpi)))


# ---------------------------------------------------------------------------
# UTM: conversioni e calcolo griglia
# ---------------------------------------------------------------------------

def utm_zone(lon: float) -> int:
    """Numero zona UTM WGS84 dalla longitudine."""
    return int((lon + 180) / 6) + 1


def get_utm_transformer(lat: float, lon: float):
    """
    Ritorna un Transformer WGS84 -> UTM per la zona che contiene (lat, lon).
    Gestisce le eccezioni norvegesi (zone 32V / 31X-37X) per completezza.
    """
    zone = utm_zone(lon)
    hem  = "north" if lat >= 0 else "south"
    proj_str = f"+proj=utm +zone={zone} +{hem} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", proj_str, always_xy=True), zone


def utm_grid_spacing(scale: int) -> int:
    """
    Spaziatura automatica della griglia UTM in metri in base alla scala nominale.
    Segue le convenzioni cartografiche standard IGM/NATO.
    """
    if scale <= 5_000:
        return 100
    elif scale <= 12_500:
        return 250
    elif scale <= 25_000:
        return 500
    elif scale <= 75_000:
        return 1_000
    elif scale <= 150_000:
        return 2_000
    elif scale <= 300_000:
        return 5_000
    else:
        return 10_000


def geo_to_map_pixel(
    lat: float, lon: float,
    center_lat: float, center_lon: float,
    zoom: int,
    map_w: int, map_h: int,
) -> tuple:
    """
    Converte (lat, lon) geografiche in pixel relativi all'angolo superiore sinistro
    dell'immagine mappa (non del canvas completo).
    """
    cx, cy = deg2global_pixel(center_lat, center_lon, zoom)
    px, py = deg2global_pixel(lat, lon, zoom)
    x = px - cx + map_w / 2
    y = py - cy + map_h / 2
    return x, y


def map_pixel_to_geo(
    px: float, py: float,
    center_lat: float, center_lon: float,
    zoom: int,
    map_w: int, map_h: int,
) -> tuple:
    """Inverso di geo_to_map_pixel: pixel mappa -> (lat, lon)."""
    cx, cy = deg2global_pixel(center_lat, center_lon, zoom)
    gx = px - map_w / 2 + cx
    gy = py - map_h / 2 + cy
    n   = 2 ** zoom
    lon = gx / (n * TILE_SIZE) * 360 - 180
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * gy / (n * TILE_SIZE))))
    lat = math.degrees(lat_r)
    return lat, lon


def compute_utm_grid_lines(
    center_lat: float, center_lon: float,
    zoom: int,
    map_w: int, map_h: int,
    scale: int,
    resize_factor: float,
) -> dict:
    """
    Calcola le linee UTM (easting e northing) da disegnare sulla mappa.

    Ritorna un dict con:
      'easting_lines'  : list of { 'E': int, 'x_pixels': list[float] (una per ogni y) }
      'northing_lines' : list of { 'N': int, 'y_pixels': list[float] }
      'spacing_m'      : spaziatura usata in metri
      'zone'           : numero zona UTM
      'utm_tf'         : Transformer usato
    """
    spacing = utm_grid_spacing(scale)
    utm_tf, zone = get_utm_transformer(center_lat, center_lon)

    # Coordinate UTM degli angoli della mappa (con un buffer del 10%)
    corners_geo = [
        map_pixel_to_geo(0,     0,      center_lat, center_lon, zoom, map_w, map_h),
        map_pixel_to_geo(map_w, 0,      center_lat, center_lon, zoom, map_w, map_h),
        map_pixel_to_geo(0,     map_h,  center_lat, center_lon, zoom, map_w, map_h),
        map_pixel_to_geo(map_w, map_h,  center_lat, center_lon, zoom, map_w, map_h),
    ]
    corners_utm = [utm_tf.transform(lo, la) for la, lo in corners_geo]  # always_xy: (lon, lat) → (E, N)

    E_vals = [c[0] for c in corners_utm]
    N_vals = [c[1] for c in corners_utm]
    E_min, E_max = min(E_vals), max(E_vals)
    N_min, N_max = min(N_vals), max(N_vals)

    E_start = math.floor(E_min / spacing) * spacing
    E_end   = math.ceil(E_max  / spacing) * spacing
    N_start = math.floor(N_min / spacing) * spacing
    N_end   = math.ceil(N_max  / spacing) * spacing

    inv_tf = Transformer.from_crs(utm_tf.target_crs, "EPSG:4326", always_xy=True)

    # --- Linee di Easting (verticali) ---
    # Per ogni valore E, campiona N lungo tutta l'altezza della mappa
    N_SAMPLES = 64
    easting_lines = []
    for E in range(int(E_start), int(E_end) + spacing, spacing):
        pts = []
        for k in range(N_SAMPLES + 1):
            N = N_min + (N_max - N_min) * k / N_SAMPLES
            try:
                lon_g, lat_g = inv_tf.transform(float(E), float(N))
                px, py = geo_to_map_pixel(lat_g, lon_g, center_lat, center_lon, zoom, map_w, map_h)
                pts.append((px * resize_factor, py * resize_factor))
            except Exception:
                continue
        if pts:
            easting_lines.append({'E': E, 'pts': pts})

    # --- Linee di Northing (orizzontali) ---
    E_SAMPLES = 64
    northing_lines = []
    for N in range(int(N_start), int(N_end) + spacing, spacing):
        pts = []
        for k in range(E_SAMPLES + 1):
            E = E_min + (E_max - E_min) * k / E_SAMPLES
            try:
                lon_g, lat_g = inv_tf.transform(float(E), float(N))
                px, py = geo_to_map_pixel(lat_g, lon_g, center_lat, center_lon, zoom, map_w, map_h)
                pts.append((px * resize_factor, py * resize_factor))
            except Exception:
                continue
        if pts:
            northing_lines.append({'N': N, 'pts': pts})

    return {
        'easting_lines':  easting_lines,
        'northing_lines': northing_lines,
        'spacing_m':      spacing,
        'zone':           zone,
        'utm_tf':         utm_tf,
    }


def draw_utm_grid(
    img: Image.Image,
    utm_data: dict,
    dpi: int,
) -> Image.Image:
    """
    Disegna le linee UTM e le etichette ai bordi della mappa.
    Le etichette mostrano sia le cifre intere (grande) che le ultime 3 cifre (piccole).
    Ritorna l'immagine con overlay RGBA.
    """
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    line_width = max(1, int(dpi / 96))

    fs_big   = max(7, int(dpi * 0.060))   # cifre intere km
    fs_small = max(5, int(dpi * 0.038))   # ultime 3 cifre

    def load_font(size, bold=False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        for base in [
            f"/usr/share/fonts/truetype/dejavu/{name}",
            f"/usr/share/fonts/dejavu/{name}",
        ]:
            try:
                return ImageFont.truetype(base, size)
            except OSError:
                pass
        return ImageFont.load_default()

    font_big   = load_font(fs_big,   bold=True)
    font_small = load_font(fs_small, bold=False)

    MARGIN = 4   # pixel di margine dai bordi per le etichette

    def clip_polyline(pts, W, H):
        """Ritorna solo i tratti del polilinea interni al rettangolo (0,0,W,H)."""
        segments = []
        seg = []
        for px, py in pts:
            inside = (0 <= px <= W) and (0 <= py <= H)
            if inside:
                seg.append((px, py))
            else:
                if seg:
                    segments.append(seg)
                    seg = []
        if seg:
            segments.append(seg)
        return segments

    def draw_label_dual(cx, cy, value_m, horiz=True):
        """
        Disegna etichetta UTM doppia (km interi grandi + ultime 3 cifre piccole).
        horiz=True -> etichetta orizzontale (per northing sui lati)
        horiz=False -> etichetta verticale (per easting sopra/sotto)
        """
        km_full  = value_m // 1000          # es. 4452
        last3    = value_m % 1000           # es. 0, 500, 250...
        txt_big  = str(km_full)
        txt_small = f"{last3:03d}" if last3 != 0 else ""

        # Dimensioni testi
        bb_big = draw.textbbox((0, 0), txt_big,   font=font_big)
        bw_big = bb_big[2] - bb_big[0]
        bh_big = bb_big[3] - bb_big[1]

        if txt_small:
            bb_sm  = draw.textbbox((0, 0), txt_small, font=font_small)
            bw_sm  = bb_sm[2]  - bb_sm[0]
            bh_sm  = bb_sm[3]  - bb_sm[1]
        else:
            bw_sm = bh_sm = 0

        pad = 2
        total_w = bw_big + (bw_sm + 1 if txt_small else 0) + pad * 2
        total_h = max(bh_big, bh_sm) + pad * 2

        tx = cx - total_w // 2
        ty = cy - total_h // 2

        # Clamp ai bordi
        tx = max(MARGIN, min(tx, W - total_w - MARGIN))
        ty = max(MARGIN, min(ty, H - total_h - MARGIN))

        draw.rectangle(
            [tx - pad, ty - pad, tx + total_w + pad, ty + total_h + pad],
            fill=UTM_LABEL_BG,
        )
        # Testo grande (km interi)
        draw.text((tx + pad, ty + pad), txt_big, fill=UTM_LABEL_COLOR, font=font_big)
        # Testo piccolo (ultime 3 cifre) in apice a destra
        if txt_small:
            draw.text(
                (tx + pad + bw_big + 1, ty + pad + bh_big - bh_sm),
                txt_small, fill=UTM_LABEL_COLOR, font=font_small,
            )

    W, H = w, h

    # --- Disegna linee Easting ---
    for line in utm_data['easting_lines']:
        segments = clip_polyline(line['pts'], W, H)
        for seg in segments:
            if len(seg) >= 2:
                draw.line(seg, fill=UTM_LINE_COLOR, width=line_width)

        # Etichette sul bordo superiore e inferiore
        # Cerca il punto della linea più vicino al bordo superiore (y=0) e inferiore (y=H)
        all_pts = line['pts']
        inside  = [(px, py) for px, py in all_pts if 0 <= px <= W and 0 <= py <= H]
        if inside:
            top_pt  = min(inside, key=lambda p: p[1])
            bot_pt  = max(inside, key=lambda p: p[1])
            # etichetta bordo superiore
            if top_pt[1] < H * 0.15:
                draw_label_dual(int(top_pt[0]), int(top_pt[1]) + fs_big + 4, line['E'])
            # etichetta bordo inferiore
            if bot_pt[1] > H * 0.85:
                draw_label_dual(int(bot_pt[0]), int(bot_pt[1]) - fs_big - 4, line['E'])

    # --- Disegna linee Northing ---
    for line in utm_data['northing_lines']:
        segments = clip_polyline(line['pts'], W, H)
        for seg in segments:
            if len(seg) >= 2:
                draw.line(seg, fill=UTM_LINE_COLOR, width=line_width)

        all_pts = line['pts']
        inside  = [(px, py) for px, py in all_pts if 0 <= px <= W and 0 <= py <= H]
        if inside:
            left_pt  = min(inside, key=lambda p: p[0])
            right_pt = max(inside, key=lambda p: p[0])
            # etichetta bordo sinistro
            if left_pt[0] < W * 0.15:
                draw_label_dual(int(left_pt[0]) + fs_big + 4, int(left_pt[1]), line['N'])
            # etichetta bordo destro
            if right_pt[0] > W * 0.85:
                draw_label_dual(int(right_pt[0]) - fs_big - 4, int(right_pt[1]), line['N'])

    return Image.alpha_composite(img.convert("RGBA"), overlay)

# ---------------------------------------------------------------------------
# Calcolo numero celle griglia
# ---------------------------------------------------------------------------

def compute_grid_dims(map_w_px: int, map_h_px: int) -> tuple:
    """
    Calcola il numero di colonne e righe della griglia in modo che
    ogni cella sia circa quadrata e grande tra 150 e 300 px.
    Restituisce (n_cols, n_rows).
    """
    target_cell = 220   # dimensione ideale cella in pixel
    n_cols = max(2, round(map_w_px / target_cell))
    n_rows = max(2, round(map_h_px / target_cell))
    n_cols = min(n_cols, 26)   # limita a 26 colonne (A-Z)
    return n_cols, n_rows

# ---------------------------------------------------------------------------
# Overlay griglia alfanumerica
# ---------------------------------------------------------------------------

def draw_grid(
    img: Image.Image,
    n_cols: int,
    n_rows: int,
    dpi: int,
) -> Image.Image:
    """
    Disegna la griglia alfanumerica (colonne A-Z, righe 1-N) sopra l'immagine.
    Usa un layer RGBA semitrasparente per le linee.
    """
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cell_w = w / n_cols
    cell_h = h / n_rows
    line_width = max(1, int(dpi / 96))

    font_size = max(8, int(min(cell_w, cell_h) * 0.10))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()

    col_labels = [chr(ord('A') + i) for i in range(n_cols)]
    row_labels  = [str(i + 1) for i in range(n_rows)]

    # Linee verticali
    for i in range(1, n_cols):
        x = int(i * cell_w)
        draw.line([(x, 0), (x, h)], fill=GRID_LINE_COLOR, width=line_width)

    # Linee orizzontali
    for j in range(1, n_rows):
        y = int(j * cell_h)
        draw.line([(0, y), (w, y)], fill=GRID_LINE_COLOR, width=line_width)

    # Etichette colonne (sopra e sotto)
    #for i, lbl in enumerate(col_labels):
    #    cx = int(i * cell_w + cell_w / 2)
    #    for cy in [int(cell_h * 0.05), int(h - cell_h * 0.05)]:
    #        _draw_label_overlay(draw, lbl, cx, cy, font)

    # Etichette righe (sinistra e destra)
    #for j, lbl in enumerate(row_labels):
    #    cy = int(j * cell_h + cell_h / 2)
    #    for cx in [int(cell_w * 0.05), int(w - cell_w * 0.05)]:
    #        _draw_label_overlay(draw, lbl, cx, cy, font)

    return Image.alpha_composite(img.convert("RGBA"), overlay)


def _draw_label_overlay(draw, text, x, y, font, padding=3):
    """Etichetta centrata con sfondo bianco su layer RGBA."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x - tw // 2
    ty = y - th // 2
    draw.rectangle(
        [tx - padding, ty - padding, tx + tw + padding, ty + th + padding],
        fill=GRID_LABEL_BG,
    )
    draw.text((tx, ty), text, fill=GRID_LABEL_COLOR, font=font)

def _draw_label_canvas(draw, text, cx, cy, font, padding=3):
    """Etichetta centrata con sfondo bianco su draw RGB diretto."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = cx - tw // 2
    ty = cy - th // 2
    draw.rectangle(
        [tx - padding, ty - padding, tx + tw + padding, ty + th + padding],
        fill="white", outline=(180, 180, 210),
    )
    draw.text((tx, ty), text, fill=(30, 30, 80), font=font)


# ---------------------------------------------------------------------------
# Download tile
# ---------------------------------------------------------------------------

class TileCache:
    """Cache in memoria: evita di riscaricare tile gia' ottenute."""

    def __init__(self):
        self._cache = {}

    def get(self, z: int, x: int, y: int) -> Image.Image:
        key = (z, x, y)
        if key in self._cache:
            return self._cache[key]

        url = TILE_URL.format(z=z, x=x, y=y)
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                self._cache[key] = img
                time.sleep(REQUEST_DELAY)
                return img
            except requests.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(f"Download tile {url} fallito: {e}")
                time.sleep(1.5)


# ---------------------------------------------------------------------------
# Composizione mappa
# ---------------------------------------------------------------------------

def build_map(
    lat: float,
    lon: float,
    zoom: int,
    width_px: int,
    height_px: int,
    cache: TileCache,
    verbose: bool = True,
) -> Image.Image:
    """Scarica e assembla le tile in un'immagine centrata su (lat, lon)."""
    cx, cy = deg2global_pixel(lat, lon, zoom)
    x0 = cx - width_px  / 2
    y0 = cy - height_px / 2

    tx0 = int(x0 // TILE_SIZE)
    ty0 = int(y0 // TILE_SIZE)
    tx1 = int((x0 + width_px  - 1) // TILE_SIZE)
    ty1 = int((y0 + height_px - 1) // TILE_SIZE)

    grid_w = tx1 - tx0 + 1
    grid_h = ty1 - ty0 + 1
    total  = grid_w * grid_h

    if total > MAX_TILES:
        raise ValueError(f"Griglia troppo grande: {total} tile ({grid_w}x{grid_h}) @ zoom {zoom}.\n"
            f"  Suggerimenti per ridurre:\n"
            f"    - Usa un formato carta più piccolo (attuale: ~{grid_w*TILE_SIZE}x{grid_h*TILE_SIZE} px download)\n"
            f"    - Riduci i DPI (es. 150 invece di valori più alti)\n"
            f"    - Usa una scala meno dettagliata (denominatore più grande)\n"
            f"  Limite attuale: MAX_TILES={MAX_TILES}"
        )

    if verbose:
        print(f"  Download {total} tile ({grid_w}x{grid_h}) @ zoom {zoom}...")

    mosaic = Image.new("RGBA", (grid_w * TILE_SIZE, grid_h * TILE_SIZE))
    count = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            count += 1
            if verbose:
                print(f"    tile {count}/{total}  ({tx},{ty})", end="\r", flush=True)
            tile = cache.get(zoom, tx, ty)
            mosaic.paste(tile, ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE))

    if verbose:
        print()

    crop_x = int(x0 - tx0 * TILE_SIZE)
    crop_y = int(y0 - ty0 * TILE_SIZE)
    return mosaic.crop((crop_x, crop_y, crop_x + width_px, crop_y + height_px))


# ---------------------------------------------------------------------------
# Layout finale: titolo, griglia, margini, scala grafica, crediti
# ---------------------------------------------------------------------------

def compose_final(
    map_img: Image.Image,
    lat: float,
    lon: float,
    zoom: int,
    scale: int,
    dpi: int,
    title: str,
    margin_mm: float = 10,
    grid: bool = True,
    n_cols: int = None,
    n_rows: int = None,
    utm_zone_num: int = None,
) -> Image.Image:
    """
    Assembla il layout completo:
      - Titolo centrato in alto
      - Mappa con griglia alfanumerica sovrapposta
      - Cornice nera
      - Etichette griglia sui 4 lati (fuori dalla mappa)
      - Scala grafica + crediti in basso
    """
    mw, mh = map_img.size

    if grid:
        if n_cols is None or n_rows is None:
            n_cols, n_rows = compute_grid_dims(mw, mh)
        col_labels = [chr(ord('A') + i) for i in range(n_cols)]
        row_labels  = [str(i + 1) for i in range(n_rows)]
    else:
        col_labels, row_labels = [], []

    # --- Font ---
    margin_px   = int(margin_mm / 25.4 * dpi)
    fs_title    = max(10, int(margin_px * 0.70))
    fs_info     = max(7,  int(margin_px * 0.42))
    fs_grid_lbl = max(7,  int(margin_px * 0.50))

    def load_font(bold=False, size=10):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        for base in [
            f"/usr/share/fonts/truetype/dejavu/{name}",
            f"/usr/share/fonts/dejavu/{name}",
        ]:
            try:
                return ImageFont.truetype(base, size)
            except OSError:
                pass
        return ImageFont.load_default()

    font_title  = load_font(bold=True,  size=fs_title)
    font_info   = load_font(bold=False, size=fs_info)
    font_glabel = load_font(bold=True,  size=fs_grid_lbl)

    # --- Spazi layout ---
    title_h   = fs_title + margin_px // 2
    glabel_sz = fs_grid_lbl + 6  if grid else 0
    bottom_h  = int(margin_px * 1.4)

    # Dimensioni canvas totale
    canvas_w = margin_px + glabel_sz + mw + glabel_sz + margin_px
    canvas_h = margin_px + title_h + glabel_sz + mh + glabel_sz + bottom_h + margin_px

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    # Posizione angolo superiore sinistro della mappa nel canvas
    map_x = margin_px + glabel_sz
    map_y = margin_px + title_h + glabel_sz

    # --- Overlay griglia sulla mappa ---
    if grid:
        map_img = draw_grid(map_img, n_cols, n_rows, dpi)

    canvas.paste(map_img.convert("RGB"), (map_x, map_y))
    draw = ImageDraw.Draw(canvas)

    # --- Cornice mappa ---
    draw.rectangle(
        [map_x - 1, map_y - 1, map_x + mw, map_y + mh],
        outline="black", width=2,
    )

    # --- Titolo ---
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(
        ((canvas_w - title_w) // 2, margin_px // 2),
        title, fill="black", font=font_title,
    )

    # --- Etichette griglia sui 4 lati ---
    if grid:
        cell_w = mw / n_cols
        cell_h = mh / n_rows

        for i, lbl in enumerate(col_labels):
            cx = map_x + int(i * cell_w + cell_w / 2)
            for gy in [map_y - glabel_sz // 2, map_y + mh + glabel_sz // 2]:
                _draw_label_canvas(draw, lbl, cx, gy, font_glabel)

        for j, lbl in enumerate(row_labels):
            cy = map_y + int(j * cell_h + cell_h / 2)
            for gx in [map_x - glabel_sz // 2, map_x + mw + glabel_sz // 2]:
                _draw_label_canvas(draw, lbl, gx, cy, font_glabel)

    # --- Scala grafica ---
    # Usa la scala nominale richiesta (non quella dello zoom grezzo)
    # per calcolare la barra: 1 px su carta = scale * (0.0254/dpi) metri reali
    mpp = scale * (0.0254 / dpi)
    bar_target_m = mpp * mw * 0.20
    magnitude  = 10 ** math.floor(math.log10(bar_target_m))
    nice_vals  = [1, 2, 2.5, 5, 10]
    bar_m  = min((v * magnitude for v in nice_vals), key=lambda v: abs(v - bar_target_m))
    bar_px = int(bar_m / mpp)
    bar_h  = max(4, int(margin_px * 0.18))

    bar_x = map_x
    bar_y = map_y + mh + glabel_sz + int(bottom_h * 0.15)

    # Barra zebrata (4 segmenti)
    seg = max(1, bar_px // 4)
    for i, color in enumerate(["black", "white", "black", "white"]):
        draw.rectangle(
            [bar_x + i * seg, bar_y, bar_x + (i + 1) * seg, bar_y + bar_h],
            fill=color, outline="black",
        )
    draw.rectangle([bar_x, bar_y, bar_x + bar_px, bar_y + bar_h], outline="black", width=1)

    label_m   = int(bar_m)
    bar_label = f"{label_m} m" if label_m < 1000 else f"{label_m / 1000:.0f} km"
    draw.text((bar_x,          bar_y + bar_h + 2), "0",        fill="black", font=font_info)
    draw.text((bar_x + bar_px, bar_y + bar_h + 2), bar_label,  fill="black", font=font_info)

    scale_text = f"1:{scale:,}"
    draw.text((bar_x, bar_y + bar_h + fs_info + 4), scale_text, fill="black", font=font_info)

    # crediti---------------------
    
    font_credits = load_font(bold=False, size=max(5, int(margin_px * 0.30)))
    
    utm_zone_str = f"  |  UTM zona {utm_zone_num}" if utm_zone_num else ""
    credits = ( f"Centro: {lat:.5f}N  {lon:.5f}E  |  Zoom: {zoom}  |  DPI: {dpi}{utm_zone_str}\n© OpenTopoMap (CC-BY-SA)  |  © OpenStreetMap contributors (ODbL)")
    credits_bbox = draw.textbbox((0, 0), credits, font=font_credits)
    credits_w = credits_bbox[2] - credits_bbox[0]
    credits_x = canvas_w - margin_px - credits_w
    draw.text( (credits_x, bar_y), credits, fill="#666666", font=font_credits)

    return canvas


# ---------------------------------------------------------------------------
# Interfaccia principale
# ---------------------------------------------------------------------------

def generate_map(
    lat: float,
    lon: float,
    scale: int        = 25_000,
    paper: str        = "A4",
    landscape: bool   = False,
    dpi: int          = 150,
    margin_mm: float  = 10,
    title: str        = None,
    grid: bool        = True,
    grid_cols: int    = None,
    grid_rows: int    = None,
    output: str       = None,
    verbose: bool     = True,
) -> Path:
    """
    Genera una mappa topografica OpenTopoMap e la salva come PNG.

    Parametri
    ----------
    lat, lon    : coordinate centro (gradi decimali)
    scale       : denominatore scala (es. 25000 -> 1:25000)
    paper       : "A0"-"A4"
    landscape   : True per orientamento orizzontale
    dpi         : risoluzione (96 / 150 / 200 / 300)
    margin_mm   : margine bianco in mm
    title       : titolo in cima alla mappa (default: coordinate)
    grid        : True per sovrapporre griglia alfanumerica
    grid_cols   : numero colonne griglia (default: automatico)
    grid_rows   : numero righe griglia (default: automatico)
    output      : percorso PNG output (default: nome automatico)
    verbose     : stampa avanzamento
    """
    paper = paper.upper()
    if paper not in PAPER_SIZES:
        raise ValueError(f"Formato '{paper}' non supportato. Scegli tra: {list(PAPER_SIZES)}")

    pw_mm, ph_mm = PAPER_SIZES[paper]
    if landscape:
        pw_mm, ph_mm = ph_mm, pw_mm

    margin_px  = int(margin_mm / 25.4 * dpi)
    fs_grid    = max(7, int(margin_px * 0.50))
    glabel_sz  = (fs_grid + 6) if grid else 0
    bottom_h   = int(margin_px * 1.4)
    fs_title   = max(10, int(margin_px * 0.70))
    title_h    = fs_title + margin_px // 2

    map_w_px = int(pw_mm / 25.4 * dpi) - 2 * margin_px - 2 * glabel_sz
    map_h_px = int(ph_mm / 25.4 * dpi) - 2 * margin_px - 2 * glabel_sz - title_h - bottom_h

    zoom, resize_factor = best_zoom_and_resize(lat, scale, dpi)
    mpp  = meters_per_pixel(lat, zoom)

    # Dimensioni da scaricare: piu' grandi se poi ridurremo (resize_factor < 1)
    download_w = int(math.ceil(map_w_px / resize_factor))
    download_h = int(math.ceil(map_h_px / resize_factor))

    if title is None:
        title = f"{lat:.4f}N  {lon:.4f}E"

    if verbose:
        orient = "landscape" if landscape else "portrait"
        n_c, n_r = compute_grid_dims(map_w_px, map_h_px) if grid else (0, 0)
        copertura_km_w = map_w_px * scale * 0.0254 / dpi / 1000
        copertura_km_h = map_h_px * scale * 0.0254 / dpi / 1000
        print(f"\n{'='*62}")
        print(f"  Mappa topografica OpenTopoMap")
        print(f"  Titolo   : {title}")
        print(f"  Centro   : {lat}N  {lon}E")
        print(f"  Scala    : 1:{scale:,} (esatta)")
        print(f"  Formato  : {paper} {orient} @ {dpi} DPI")
        if zoom == 17 and resize_factor > 1.0:
            print(f"  ⚠  Zoom cappato a 17 (max OpenTopoMap): upscale {resize_factor:.3f}x — qualita' ridotta per scale molto dettagliate")
        print(f"  Zoom OSM : {zoom}  resize: {resize_factor:.4f}x")
        print(f"  Download : {download_w}x{download_h} px  ->  output {map_w_px}x{map_h_px} px")
        print(f"  Copertura: {copertura_km_w:.1f} x {copertura_km_h:.1f} km")
        if grid:
            print(f"  Griglia  : {grid_cols or n_c} col x {grid_rows or n_r} righe")
        print(f"{'='*62}")

    cache = TileCache()
    # Scarica la mappa a dimensione maggiorata
    mappa_raw = build_map(lat, lon, zoom, download_w, download_h, cache, verbose)
    # Ricampiona alla dimensione finale: ottiene scala esatta
    if abs(resize_factor - 1.0) > 0.001:
        mappa = mappa_raw.resize((map_w_px, map_h_px), Image.LANCZOS)
        if verbose:
            print(f"  Ricampionamento {download_w}x{download_h} -> {map_w_px}x{map_h_px} (scala esatta 1:{scale:,})")
    else:
        mappa = mappa_raw

    n_cols_eff = grid_cols
    n_rows_eff = grid_rows
    if grid and (n_cols_eff is None or n_rows_eff is None):
        n_cols_eff, n_rows_eff = compute_grid_dims(map_w_px, map_h_px)

    # --- Griglia UTM ---
    utm_data = compute_utm_grid_lines(
        center_lat   = lat,
        center_lon   = lon,
        zoom         = zoom,
        map_w        = download_w,
        map_h        = download_h,
        scale        = scale,
        resize_factor = resize_factor,
    )
    if verbose:
        print(f"  Griglia UTM : zona {utm_data['zone']}  |  passo {utm_data['spacing_m']} m  "
              f"|  {len(utm_data['easting_lines'])} E + {len(utm_data['northing_lines'])} N linee")
    mappa = draw_utm_grid(mappa, utm_data, dpi)

    finale = compose_final(
        map_img      = mappa,
        lat          = lat,
        lon          = lon,
        zoom         = zoom,
        scale        = scale,
        dpi          = dpi,
        title        = title,
        margin_mm    = margin_mm,
        grid         = grid,
        n_cols       = n_cols_eff,
        n_rows       = n_rows_eff,
        utm_zone_num = utm_data['zone'],
    )
    
    #MODIFICARE TITOLO FILE
    if output is None:
        lat_s  = f"{abs(lat):.4f}{'N' if lat >= 0 else 'S'}"
        lon_s  = f"{abs(lon):.4f}{'E' if lon >= 0 else 'W'}"
        
        #output pdf
        #output = f"topo_{lat_s}_{lon_s}_1-{scale}_{paper}_{dpi}dpi.pdf"
        output=f"{title}-1_{scale}-{paper}-{dpi}.pdf"
        out_path = Path(output)
        finale.save(out_path, "pdf", dpi=(dpi, dpi))
        
        #ext_file=["pdf", "svg", "png"]
        #for ext in ext_file:
        #    output=f"{title}-1_{scale}-{paper}-{dpi}.{ext}"
        #    out_path = Path(output)
        #    finale.save(out_path, "pdf", dpi=(dpi, dpi))

        if verbose:
            size_mb = out_path.stat().st_size / 1024**2
            print(f"\n  Mappa salvata: {out_path}  ({size_mb:.1f} MB)")

        return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    print("\n--- TopoMap Generator: Configurazione Guidata ---")
    
    place_file= open('luoghi.csv', 'r', encoding='utf-8')
    reader=csv.reader(place_file)
    places=[]
    for row in reader:
        places.append(row)
    
    places=places[1:]
    
    try:
        for row in places:
            title=row[0]
            lat=float(row[1])
            lon=float(row[2])
            scale=int(row[3])
            #scale=1000
            dpi=int(row[4])
            #dpi=300
            paper=row[5]
            #paper="A4"
            landscape=True if row[6]=="y" else False
            #landscape=True
        
            generate_map(
                lat        = lat,
                lon        = lon,
                scale      = scale,
                paper      = paper,
                landscape  = landscape,
                dpi        = dpi,
                title      = title,
                grid       = True,
                verbose    = True)
        
    except ValueError as e:
        print(f"\nErrore nell'inserimento dei dati: {e}")
    except KeyboardInterrupt:
        print("\n\nOperazione annullata dall'utente.")

if __name__ == "__main__":
    main()  
