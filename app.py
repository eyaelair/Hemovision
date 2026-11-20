"""HemoVision | Educational conjunctival image-analysis studio.

Run with: streamlit run HemoVision_REPAIRED_2026.py
No anemia diagnosis, screening classification, or hemoglobin prediction is performed.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title='HemoVision | Imaging Studio', page_icon='H', layout='wide', initial_sidebar_state='collapsed')

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#e9eff7; --muted:#9aabbc; --line:#23334a; --accent:#69e1cd; --bg:#0a111d; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse 900px 470px at 74% -16%,#123d4c 0%,transparent 72%), #0a111d;color:var(--ink);}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#101b2b;border-right:1px solid #26364b;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {font-family:'Manrope',sans-serif;}
.block-container { max-width:1360px; padding-top:2.15rem; padding-bottom:4rem; }
h1,h2,h3 {font-family:'Manrope',sans-serif !important;letter-spacing:-.045em !important;}
h1 {font-weight:800 !important;} h2 {font-weight:750 !important;} h3 {font-weight:700 !important;}
p,li {line-height:1.6;}
.hero {position:relative;overflow:hidden;background:linear-gradient(105deg,#132838 0%,#132337 60%,#173d45 100%);border:1px solid #294656;border-radius:22px;padding:36px 40px;margin:0 0 24px;}
.hero:after {content:'';position:absolute;right:-75px;top:-130px;width:340px;height:340px;background:radial-gradient(circle,rgba(105,225,205,.15),transparent 67%);pointer-events:none;}
.eyebrow {font-size:.68rem;letter-spacing:.18em;color:#77dbc9;font-weight:800;text-transform:uppercase;margin-bottom:12px;}
.hero h1 {font-size:3.0rem;margin:0 0 8px;font-weight:800;color:#f5fbff;line-height:1.12;}
.hero p {font-size:1rem;color:#b9cbd6;max-width:740px;margin:0;}
.pill {display:inline-block;border:1px solid #3c6c72;background:#173b46;color:#9ce4d7;border-radius:999px;padding:5px 12px;font-size:.72rem;font-weight:700;margin-top:18px;}
.section-lead {color:#9fb2c4;font-size:.92rem;margin:-8px 0 18px;}
.small-label {font-size:.73rem;color:#8eacbd;letter-spacing:.12em;font-weight:800;text-transform:uppercase;}
.note {background:#12283a;border:1px solid #264559;padding:16px 20px;border-radius:12px;color:#c4dbe5;font-size:.9rem;margin-top:14px;}
.footer {border-top:1px solid #2d3948;padding-top:18px;color:#8497ab;font-size:.8rem;margin-top:40px;}
[data-testid="stMetric"] {background:#111f30;border:1px solid #253b50;padding:14px 16px;border-radius:14px;}
[data-testid="stMetricLabel"] {color:#a7bccd;}
[data-testid="stMetricValue"] {color:#f5f9ff; font-family:'Manrope',sans-serif;}
[data-testid="stTabs"] button {font-weight:700;letter-spacing:.01em;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background-color:#69e1cd;}
[data-testid="stFileUploader"] {border-radius:12px;}
.stButton > button[kind="primary"] {background:#6be2cf !important;color:#0c2028 !important;border:none !important;font-weight:800 !important;}
.stDownloadButton > button {border:1px solid #407883 !important;}
hr {border-color:#243648;}
@media(max-width:720px) {.hero{padding:24px 22px}.hero h1{font-size:2.05rem}.block-container{padding-top:1rem}}
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)


def html_section(title: str, description: str = '') -> None:
    st.subheader(title)
    if description:
        st.markdown(f'<div class="section-lead">{description}</div>', unsafe_allow_html=True)


def image_to_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def enhance_clahe(rgb: np.ndarray, clip_limit: float) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    light, a, b = cv2.split(lab)
    processed = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((processed, a, b)), cv2.COLOR_LAB2RGB)


def calculate_quality(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    overexposed = float(np.mean(gray >= 245) * 100)
    underexposed = float(np.mean(gray <= 15) * 100)
    # Exploratory, content/resolution-sensitive warnings; not validated cutoffs.
    warnings = []
    if sharpness < 50:
        warnings.append('Potential blur: try stabilizing the camera and improving focus.')
    if brightness < 60:
        warnings.append('Low brightness: capture with more even lighting.')
    elif brightness > 220:
        warnings.append('High brightness: reduce illumination or exposure.')
    if overexposed > 10:
        warnings.append('Potential highlight clipping: reduce glare and reflections.')
    if underexposed > 10:
        warnings.append('Potential shadow clipping: increase even illumination.')
    return dict(sharpness=sharpness, brightness=brightness, overexposed=overexposed,
                underexposed=underexposed, warnings=warnings)


def calc_color(rgb: np.ndarray, mask: np.ndarray) -> dict | None:
    chosen = mask > 0
    count = int(np.count_nonzero(chosen))
    if count == 0:
        return None
    pixels = rgb[chosen].astype(np.float64)
    mean = pixels.mean(axis=0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[chosen].astype(np.float64)
    red_fraction = float(mean[0] / mean.sum()) if mean.sum() else 0.0
    return {
        'selected_pixels': count,
        'coverage_percent': 100 * count / mask.size,
        'mean_red': float(mean[0]),
        'mean_green': float(mean[1]),
        'mean_blue': float(mean[2]),
        'lab_a_8bit': float(lab[:, 1].mean()),
        'lab_l_8bit': float(lab[:, 0].mean()),
        'red_fraction': red_fraction,
    }


def image_overlaid(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    flag = mask > 0
    tint = np.array([60, 226, 196], dtype=float)
    out[flag] = np.clip(.62 * out[flag].astype(float) + .38 * tint, 0, 255).astype('uint8')
    # Exact polygon/selection outline improves visibility on light background.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (8, 110, 125), thickness=2)
    return out


def make_histogram(rgb: np.ndarray, mask: np.ndarray):
    data = rgb[mask > 0]
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor='#111f30')
    ax.set_facecolor('#111f30')
    for i, (name, color) in enumerate([('Red', '#fa7780'), ('Green', '#6be2cf'), ('Blue', '#88aaff')]):
        ax.hist(data[:, i], bins=32, range=(0, 256), histtype='step', linewidth=2,
                color=color, label=name)
    ax.set_xlim(0, 255)
    ax.set_xlabel('Pixel intensity (0–255)', color='#c8d6e3')
    ax.set_ylabel('Pixel count', color='#c8d6e3')
    ax.tick_params(colors='#a6b8c9')
    ax.grid(alpha=.12, color='white')
    for spine in ax.spines.values(): spine.set_color('#314558')
    ax.legend(frameon=False, labelcolor='#e1e8ef', ncol=3)
    fig.tight_layout()
    return fig


st.markdown('''<div class="hero"><div class="eyebrow">Research prototype / Computer vision laboratory</div>
<h1>HemoVision<span style="color:#6be2cf">.</span></h1>
<p>An explainable imaging studio for investigating visible colour characteristics of the lower-eyelid conjunctiva. Explore every step from image enhancement to pixel-level analysis.</p>
<span class="pill">EDUCATIONAL USE ONLY &nbsp; · &nbsp; NO MEDICAL DIAGNOSIS</span></div>''', unsafe_allow_html=True)

st.markdown('<div class="note"><b>Clinical limitation.</b> This tool does not diagnose or rule out anemia, estimate hemoglobin, or determine whether medical testing is necessary. A clinician and laboratory tests are needed for diagnosis. Avoid uploading identifiable images without consent.</div>', unsafe_allow_html=True)

# Main-screen upload: visible even if the sidebar is collapsed.
st.markdown('''<div style="margin-top:1.75rem;margin-bottom:.4rem">
<span class="small-label">01 / GET STARTED</span>
<h2 style="font-size:1.55rem;margin:.35rem 0 .25rem;color:#eef6ff">Upload an eye photograph</h2>
<p style="margin:0;color:#aec3d4">Select a JPEG or PNG showing the exposed inner lower eyelid.</p>
</div>''', unsafe_allow_html=True)
with st.container(border=True):
    uploaded = st.file_uploader(
        'Choose a conjunctival photograph',
        type=['jpg', 'jpeg', 'png'],
        help='Use only photographs that you are permitted to analyse.',
        key='home_image_upload',
    )
    if uploaded is None:
        st.caption('Drag and drop an image here or choose Browse files. Your image is analysed locally by this Streamlit app.')
    else:
        st.success('Image received. Set your crop below, then choose the tissue in the first tab.')

with st.sidebar:
    st.markdown('<div class="small-label">HEMOVISION / CONTROLS</div>', unsafe_allow_html=True)
    st.header('Imaging workspace')
    st.divider()
    st.subheader('Preprocessing')
    clip_limit = st.slider('CLAHE contrast limit', 1.0, 5.0, 2.0, 0.5)
    st.subheader('Colour segmentation')
    hue_max = st.slider('Red hue upper range', 0, 25, 15)
    saturation_min = st.slider('Minimum saturation', 0, 255, 35)
    kernel_size = st.slider('Morphology kernel', 3, 11, 5, 2)
    use_hsv_with_polygon = st.toggle('Apply HSV mask inside polygon', value=False,
                                    help='Off: all polygon pixels are included. On: exclude polygon pixels that fail HSV segmentation. Pale tissue may be filtered out.')
    st.divider()
    st.caption('Selection limits and colour thresholds are exploratory, not clinically validated.')
    st.caption('Built with Streamlit · OpenCV · NumPy')

if uploaded is None:
    left, right = st.columns([1.4, 1])
    with left:
        html_section('How HemoVision works', 'After uploading a photo, follow the guided steps to examine its visible features.')
        st.info('Use the upload box above to activate the analysis workspace.')
        st.markdown('**Workflow**  ·  Upload → Crop → Polygon → Analyse → Export')
    with right:
        st.markdown('''<div class="note"><span class="small-label">WHAT YOU WILL GET</span><br><br>
        <b>Image processing</b><br>CLAHE, HSV/LAB conversion, morphology and segmentation.<br><br>
        <b>Explainable measurements</b><br>Selected area, RGB/LAB values, redness fraction and pixel histograms.<br><br>
        <b>Image quality</b><br>Exploratory focus, brightness and exposure warnings.</div>''', unsafe_allow_html=True)
    st.stop()

try:
    original_pil = ImageOps.exif_transpose(Image.open(uploaded)).convert('RGB')
except (UnidentifiedImageError, OSError, ValueError) as exc:
    st.error(f'Unable to read this image: {exc}')
    st.stop()

if original_pil.width < 60 or original_pil.height < 60:
    st.error('Image is too small. Please use a photograph at least 60 × 60 pixels.')
    st.stop()

# Resize large image safely for faster reruns. Always disclose the effective analysis size.
original_size = original_pil.size
original_pil.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
rgb = np.asarray(original_pil)
height, width = rgb.shape[:2]
file_hash = hashlib.sha1(uploaded.getvalue()).hexdigest()[:16]

st.markdown('<div class="small-label">ACTIVE STUDY</div>', unsafe_allow_html=True)
st.write(f'**{uploaded.name}**  ·  Analysis resolution {width} × {height} px' + (f'  ·  Original {original_size[0]} × {original_size[1]} px' if original_size != original_pil.size else ''))

# Region selectors are in one place to keep the tabbed pipeline consistent.
with st.expander('1 / Rectangular crop — adjust your initial region', expanded=True):
    st.caption('Crop around the lower eyelid tissue; exclude as much iris, surrounding skin, and fingers as possible.')
    ax, ay = st.columns(2)
    with ax:
        x_start = st.slider('Left (%)', 0, 98, 20)
        y_start = st.slider('Top (%)', 0, 98, 55)
    with ay:
        x_end = st.slider('Right (%)', x_start + 1, 100, max(x_start + 1, 80))
        y_end = st.slider('Bottom (%)', y_start + 1, 100, max(y_start + 1, 85))
    x1, x2 = int(width*x_start/100), min(width, int(width*x_end/100))
    y1, y2 = int(height*y_start/100), min(height, int(height*y_end/100))
    roi = rgb[y1:y2, x1:x2].copy()
    if roi.size == 0:
        st.error('The selected crop is empty. Adjust the boundaries.')
        st.stop()
    marked = rgb.copy()
    cv2.rectangle(marked, (x1, y1), (x2 - 1, y2 - 1), (105, 225, 205), max(2, width//500))
    crop_col, roi_col = st.columns(2)
    crop_col.image(marked, caption='Location on original image', use_container_width=True)
    roi_col.image(roi, caption='Selected crop', use_container_width=True)

# Derived image products. Original ROI, not enhanced image, feeds colour metrics.
clahe_rgb = enhance_clahe(roi, clip_limit)
hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
mask_low = cv2.inRange(hsv, np.array([0, saturation_min, 40]), np.array([hue_max, 255, 255]))
mask_high = cv2.inRange(hsv, np.array([170, saturation_min, 40]), np.array([179, 255, 255]))
segmentation_mask = cv2.bitwise_or(mask_low, mask_high)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
opened_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel)
cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
quality = calculate_quality(roi)

# Selection state is tied to actual cropped pixels, not to CLAHE/HSV control changes.
roi_hash = hashlib.sha1(roi.tobytes()).hexdigest()[:20]
if st.session_state.get('active_roi_hash') != roi_hash:
    st.session_state.active_roi_hash = roi_hash
    st.session_state.polygon_points = []
    st.session_state.polygon_finished = False
    st.session_state.polygon_event = None
    st.session_state.polygon_version = st.session_state.get('polygon_version', 0) + 1


tab_select, tab_process, tab_analysis, tab_export = st.tabs([
    '01  SELECT REGION', '02  PROCESSING LAB', '03  COLOUR ANALYSIS', '04  EXPORT & REVIEW'
])

with tab_select:
    html_section('Precision polygon selection', 'Click points around the visible conjunctival tissue. Finish to create a reproducible region mask.')
    st.caption('For better boundary accuracy, use about 6–12 points. Image click positions are stored only in the current Streamlit session.')
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button('Undo last point', use_container_width=True):
            if st.session_state.polygon_points:
                st.session_state.polygon_points.pop()
            st.session_state.polygon_finished = False
            st.session_state.polygon_version += 1
            st.rerun()
    with c2:
        if st.button('Clear polygon', use_container_width=True):
            st.session_state.polygon_points = []
            st.session_state.polygon_finished = False
            st.session_state.polygon_event = None
            st.session_state.polygon_version += 1
            st.rerun()
    with c3:
        if st.button('Finish selection', use_container_width=True, type='primary'):
            if len(st.session_state.polygon_points) >= 3:
                st.session_state.polygon_finished = True
                st.rerun()
            else:
                st.warning('Select at least three distinct points first.')

    roi_h, roi_w = roi.shape[:2]
    display_w = min(720, roi_w)
    display_h = max(1, round(roi_h * display_w / roi_w))
    display_pil = Image.fromarray(roi).resize((display_w, display_h), Image.Resampling.LANCZOS)
    annotated = display_pil.copy()
    artist = ImageDraw.Draw(annotated)
    pts = st.session_state.polygon_points
    if len(pts) > 1:
        artist.line(pts, fill='#69e1cd', width=3)
    if st.session_state.polygon_finished and len(pts) >= 3:
        artist.line([pts[-1], pts[0]], fill='#69e1cd', width=3)
    for px, py in pts:
        artist.ellipse((px-5, py-5, px+5, py+5), fill='#eafffb', outline='#0f7071', width=2)

    if st.session_state.polygon_finished:
        st.image(annotated, caption='Final polygon. Clear or undo to edit.', width=display_w)
    else:
        click = streamlit_image_coordinates(annotated,
            key=f'polygon_{roi_hash}_{st.session_state.polygon_version}', cursor='crosshair')
        if click is not None:
            # Component includes unix_time, allowing a repeated rerun to be ignored.
            event = (click.get('x'), click.get('y'), click.get('unix_time'))
            if event != st.session_state.polygon_event:
                st.session_state.polygon_event = event
                cx = int(np.clip(click['x'], 0, display_w-1))
                cy = int(np.clip(click['y'], 0, display_h-1))
                st.session_state.polygon_points.append((cx, cy))
                st.rerun()

    st.caption(f'{len(pts)} polygon vertices selected' + ('  ·  Selection finished' if st.session_state.polygon_finished else '  ·  Awaiting clicks'))

# Generate polygon mask AFTER click processing, for all tabs.
polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
if st.session_state.polygon_finished and len(st.session_state.polygon_points) >= 3:
    vertices = np.array([[min(roi_w-1,int(round(px*roi_w/display_w))),
                          min(roi_h-1,int(round(py*roi_h/display_h)))]
                         for px, py in st.session_state.polygon_points], dtype=np.int32)
    cv2.fillPoly(polygon_mask, [vertices], 255)
    if np.count_nonzero(polygon_mask) < 10:
        st.warning('Selected polygon is very small. Try a larger tissue area.')

analysis_mask = cv2.bitwise_and(polygon_mask, cleaned_mask) if use_hsv_with_polygon else polygon_mask.copy()
metrics = calc_color(roi, analysis_mask)

with tab_select:
    if st.session_state.polygon_finished:
        html_section('Selection review', 'The overlay shows precisely which original-image pixels will be included in the analysis.')
        s1, s2, s3 = st.columns(3)
        s1.image(polygon_mask, caption='Polygon mask', use_container_width=True)
        s2.image(analysis_mask, caption='Final analysis mask', use_container_width=True)
        s3.image(image_overlaid(roi, analysis_mask), caption='Selected tissue overlay', use_container_width=True)
        if metrics:
            st.metric('Analysed area of crop', f"{metrics['coverage_percent']:.1f}%")
        else:
            st.warning('No pixels passed the selection. Disable HSV filtering or adjust the thresholds.')
        if use_hsv_with_polygon:
            st.caption('HSV filtering is ON. Pale tissue can be excluded even when it is inside your polygon.')
        else:
            st.caption('HSV filtering is OFF. All pixels inside the polygon are used; the HSV mask is still available for comparison in the Processing Lab.')
    else:
        st.info('Finish a polygon to unlock quantitative colour analysis and downloads.')

with tab_process:
    html_section('Image processing laboratory', 'Inspect what each technique changes. The application calculates colour metrics from the original pixels.')
    st.markdown('**A / Contrast-limited adaptive histogram equalization**')
    p1, p2 = st.columns(2)
    p1.image(roi, caption='Original crop', use_container_width=True)
    p2.image(clahe_rgb, caption=f'CLAHE enhanced · clip limit {clip_limit:.1f}', use_container_width=True)
    st.markdown('**B / Colour spaces**')
    p1, p2, p3 = st.columns(3)
    p1.image(roi, caption='RGB original', use_container_width=True)
    p2.image(hsv[:, :, 1], caption='HSV saturation channel', use_container_width=True, clamp=True)
    p3.image(lab[:, :, 1], caption='LAB A channel (8-bit)', use_container_width=True, clamp=True)
    st.markdown('**C / Colour thresholding and morphology**')
    p1, p2, p3 = st.columns(3)
    p1.image(segmentation_mask, caption='HSV binary mask', use_container_width=True)
    p2.image(cleaned_mask, caption='After opening and closing', use_container_width=True)
    p3.image(cv2.bitwise_and(roi, roi, mask=cleaned_mask), caption='HSV-selected colour regions', use_container_width=True)
    st.caption('HSV red ranges are a demonstration of threshold-based segmentation; they are not a reliable anatomical detector and can select skin or miss pale conjunctiva.')

with tab_analysis:
    html_section('Colour analysis', 'Pixel-level measurements from the final region selected in the first tab.')
    if metrics is None:
        st.info('Complete a nonempty polygon in “Select Region” to calculate colour metrics.')
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Mean R', f"{metrics['mean_red']:.1f}")
        m2.metric('Mean G', f"{metrics['mean_green']:.1f}")
        m3.metric('Mean B', f"{metrics['mean_blue']:.1f}")
        m4.metric('LAB A (8-bit)', f"{metrics['lab_a_8bit']:.1f}")
        n1, n2, n3 = st.columns(3)
        n1.metric('Red fraction', f"{metrics['red_fraction']:.3f}")
        n2.metric('Analysed crop area', f"{metrics['coverage_percent']:.1f}%")
        n3.metric('Selected pixels', f"{metrics['selected_pixels']:,}")
        st.caption('Red fraction = mean(R) / [mean(R)+mean(G)+mean(B)]. It is NOT a validated anemia metric or the published eMoglobin method. LAB A is OpenCV 8-bit encoded (neutral near 128).')
        st.image(image_overlaid(roi, analysis_mask), caption='Analysed tissue location', use_container_width=True)
        fig = make_histogram(roi, analysis_mask)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('<div class="note">Colour differences can arise from lighting, white balance, camera processing, pigmentation and the selected region. These figures cannot determine hemoglobin levels or whether someone has anemia.</div>', unsafe_allow_html=True)

with tab_export:
    html_section('Quality review & exports', 'Use this section to document your processing workflow and decide whether a photograph should be retaken.')
    q1, q2, q3, q4 = st.columns(4)
    q1.metric('Sharpness (heuristic)', f"{quality['sharpness']:.1f}")
    q2.metric('Mean brightness', f"{quality['brightness']:.1f}/255")
    q3.metric('Overexposed', f"{quality['overexposed']:.1f}%")
    q4.metric('Underexposed', f"{quality['underexposed']:.1f}%")
    if quality['warnings']:
        st.warning('Quality review suggests retaking or adjusting this image:\n\n' + '\n\n'.join('- ' + w for w in quality['warnings']))
    else:
        st.success('No major issues detected by these basic image-quality heuristics.')
    st.caption('Quality thresholds are illustrative and resolution/content dependent. They are not validated clinical acceptance criteria.')
    st.divider()
    if metrics is None:
        st.info('Complete the polygon to enable exporting measurements and the mask.')
    else:
        st.markdown('**Export analysis package**')
        report = {
            'project': 'HemoVision',
            'purpose': 'Educational image analysis only; not an anemia diagnosis',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'file_name': uploaded.name,
            'image_resolution': [width, height],
            'crop_percent': dict(left=x_start, right=x_end, top=y_start, bottom=y_end),
            'polygon_vertices_display': st.session_state.polygon_points,
            'hsv_filter_inside_polygon': use_hsv_with_polygon,
            'hsv_settings': dict(hue_max=hue_max, min_saturation=saturation_min, morphology_kernel=kernel_size),
            'colour_metrics': metrics,
            'quality_metrics': quality,
            'limitations': ['Not clinically validated', 'No hemoglobin estimate', 'No anemia classification', 'RGB values sensitive to acquisition conditions'],
        }
        csv_data = pd.DataFrame([metrics]).to_csv(index=False).encode('utf-8')
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('analysis.json', json.dumps(report, indent=2))
            z.writestr('colour_metrics.csv', csv_data)
            z.writestr('crop_original.png', image_to_png(roi))
            z.writestr('enhanced_clahe.png', image_to_png(clahe_rgb))
            z.writestr('polygon_mask.png', image_to_png(polygon_mask))
            z.writestr('analysis_mask.png', image_to_png(analysis_mask))
            z.writestr('selected_tissue_overlay.png', image_to_png(image_overlaid(roi, analysis_mask)))
        e1, e2 = st.columns(2)
        e1.download_button('Download complete analysis (.zip)', bundle.getvalue(),
                           file_name='hemovision_analysis.zip', mime='application/zip', use_container_width=True, type='primary')
        e2.download_button('Download measurements (.csv)', csv_data,
                           file_name='hemovision_metrics.csv', mime='text/csv', use_container_width=True)
        with st.expander('View machine-readable analysis summary'):
            st.json(report)
    st.markdown('''<div class="note"><b>Suggested user decision:</b> If the selected region is inaccurate, redraw it. If blur, shadow, or glare is severe, retake the photograph. Medical decisions require professional evaluation and appropriate testing, not these numerical outputs.</div>''', unsafe_allow_html=True)

st.markdown('<div class="footer">HEMOVISION &nbsp; / &nbsp; AIT204 Computer Vision &nbsp; / &nbsp; Research demonstration only &nbsp; · &nbsp; No diagnostic claims</div>', unsafe_allow_html=True)
