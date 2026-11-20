import csv
import importlib.util
import re
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="FarmaCheck IA",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "mobilenetv2_farmacos.keras"
FAKE_LOTS_PATH = ROOT / "data" / "lotes_falsificados.csv"

st.markdown(
    """
    <style>
    .block-container {max-width: 850px; padding-top: 1.6rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: .8rem; border-radius: .7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💊 FarmaCheck IA")
st.write("**Evaluación preliminar de empaques farmacéuticos**")
st.caption("Prototipo académico · Visión por computadora + OCR")

st.warning(
    "Esta herramienta es experimental. Analiza únicamente señales visibles del empaque; "
    "no comprueba la composición química, no certifica autenticidad y no sustituye la "
    "verificación oficial de DIGEMID."
)

with st.expander("¿Cómo funciona?", expanded=False):
    st.write(
        "1. Captura o selecciona una fotografía del empaque.\n"
        "2. El sistema revisa la calidad básica de la imagen.\n"
        "3. La CNN analizará características visuales y el OCR leerá los datos impresos "
        "cuando esos módulos estén entrenados e instalados.\n"
        "4. Las reglas buscarán inconsistencias y el sistema mostrará la evidencia disponible."
    )

def assess_image_quality(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    issues = []
    if sharpness < 45:
        issues.append("La imagen podría estar desenfocada.")
    if brightness < 45:
        issues.append("La imagen parece demasiado oscura.")
    elif brightness > 220:
        issues.append("La imagen podría estar sobreexpuesta.")
    return sharpness, brightness, issues

def run_ocr(rgb: np.ndarray):
    if importlib.util.find_spec("easyocr") is None:
        return [], "El módulo OCR todavía no está habilitado en esta primera publicación."
    try:
        import easyocr
        if "ocr_reader" not in st.session_state:
            with st.spinner("Inicializando OCR. La primera ejecución puede tardar..."):
                st.session_state.ocr_reader = easyocr.Reader(["es", "en"], gpu=False)
        with st.spinner("Leyendo los textos del empaque..."):
            results = st.session_state.ocr_reader.readtext(rgb, detail=1, paragraph=False)
        return [
            {"text": str(item[1]), "confidence": float(item[2])}
            for item in results
        ], None
    except Exception as exc:
        return [], f"El módulo OCR no pudo ejecutarse: {exc}"

def extract_fields(text: str):
    upper = text.upper()
    batch_patterns = [
        r"(?:LOTE|LOT|L)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,})",
        r"(?:SERIE)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,})",
    ]
    expiration_patterns = [
        r"(?:VENCE|VENCIMIENTO|EXP|CAD|F\.?\s*VENC)\s*[:#\-]?\s*"
        r"((?:0?[1-9]|1[0-2])[\-/](?:20)?\d{2}|"
        r"(?:0?[1-9]|[12]\d|3[01])[\-/](?:0?[1-9]|1[0-2])[\-/]20\d{2})",
    ]
    registration_patterns = [
        r"(?:REG(?:ISTRO)?\.?\s*SANITARIO|R\.?\s*S\.?)\s*[:#\-]?\s*"
        r"([A-Z]{0,4}[\-]?[A-Z]{0,3}\s*\d{3,8}[A-Z0-9\-]*)"
    ]

    def first_match(patterns):
        for pattern in patterns:
            match = re.search(pattern, upper, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" .,:;")
        return None

    return {
        "lote": first_match(batch_patterns),
        "vencimiento": first_match(expiration_patterns),
        "registro": first_match(registration_patterns),
    }

def parse_expiration(value: str):
    if not value:
        return None
    value = value.strip().replace("-", "/")
    formats = ["%m/%Y", "%m/%y", "%d/%m/%Y", "%d/%m/%y"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt).date()
            if fmt in ("%m/%Y", "%m/%y"):
                from calendar import monthrange
                return date(parsed.year, parsed.month, monthrange(parsed.year, parsed.month)[1])
            return parsed
        except ValueError:
            continue
    return None

def read_fake_lots():
    if not FAKE_LOTS_PATH.exists():
        return set()
    try:
        with FAKE_LOTS_PATH.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "lote" not in reader.fieldnames:
                return set()
            return {
                str(row["lote"]).strip().upper()
                for row in reader
                if row.get("lote")
            }
    except (OSError, csv.Error):
        return set()

@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)

def get_cnn_probability(rgb: np.ndarray):
    if not MODEL_PATH.exists():
        return None, "La CNN todavía no está entrenada. Se incorporará después del entrenamiento en Google Colab."
    try:
        import tensorflow as tf
        model = load_cnn_model(str(MODEL_PATH))
        resized = cv2.resize(rgb, (224, 224)).astype("float32")
        image_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        probabilities = model.predict(np.expand_dims(image_tensor, axis=0), verbose=0)[0]
        return float(probabilities[1]), None
    except Exception as exc:
        return None, f"La CNN no pudo ejecutarse: {exc}"

def analyze_image(rgb: np.ndarray):
    sharpness, brightness, quality_issues = assess_image_quality(rgb)

    st.subheader("1. Calidad de la fotografía")
    q1, q2 = st.columns(2)
    q1.metric("Nitidez (Laplaciano)", f"{sharpness:.1f}")
    q2.metric("Brillo promedio", f"{brightness:.1f}/255")
    if quality_issues:
        for issue in quality_issues:
            st.warning(issue)
        st.info("Intenta fotografiar el empaque de frente, con buena iluminación y sin reflejos.")
    else:
        st.success("No se detectaron problemas básicos de nitidez o iluminación.")

    st.divider()
    st.subheader("2. Lectura de datos del empaque (OCR)")
    ocr_lines, ocr_error = run_ocr(rgb)
    if ocr_error:
        st.info(ocr_error)

    extracted_text = "\n".join(line["text"] for line in ocr_lines)
    if extracted_text:
        with st.expander("Ver texto reconocido"):
            st.text(extracted_text)

    fields = extract_fields(extracted_text)
    c1, c2, c3 = st.columns(3)
    c1.metric("Lote", fields["lote"] or "No identificado")
    c2.metric("Vencimiento", fields["vencimiento"] or "No identificado")
    c3.metric("Registro sanitario", fields["registro"] or "No identificado")

    st.divider()
    st.subheader("3. Clasificación visual (CNN)")
    cnn_probability, cnn_error = get_cnn_probability(rgb)
    if cnn_probability is not None:
        st.metric("Probabilidad visual de empaque sospechoso", f"{cnn_probability:.1%}")
    else:
        st.info(cnn_error)

    if not extracted_text:
        st.info("Las reglas basadas en OCR se activarán cuando se incorpore el motor de lectura.")
    else:
        st.subheader("4. Reglas preliminares")
        active_rules = []
        weights = {"R1": 1.0, "R2": 0.8, "R3": 0.4, "R4": 1.0}

        expiration_date = parse_expiration(fields["vencimiento"])
        if expiration_date and expiration_date < date.today():
            active_rules.append(("R1", "La fecha de vencimiento reconocida ya pasó.", weights["R1"]))
        if not fields["lote"] or not fields["vencimiento"]:
            active_rules.append(("R2", "No se pudo identificar el lote o vencimiento.", weights["R2"]))
        if not fields["registro"] or not fields["lote"]:
            active_rules.append(("R3", "Uno o más campos importantes no pudieron leerse completamente.", weights["R3"]))

        fake_lots = read_fake_lots()
        if fields["lote"] and fields["lote"].upper() in fake_lots:
            active_rules.append(("R4", "El lote aparece en la lista local de lotes reportados como falsificados.", weights["R4"]))

        if active_rules:
            for code, explanation, _ in active_rules:
                st.write(f"- **{code}:** {explanation}")
        else:
            st.success("No se activaron las reglas locales disponibles. Esto no demuestra autenticidad.")

        if fields["registro"]:
            st.info("El registro extraído debe verificarse manualmente en las fuentes oficiales de DIGEMID.")

        ocr_score = sum(item[2] for item in active_rules) / sum(weights.values())
        if cnn_probability is not None:
            risk = 0.5 * cnn_probability + 0.5 * ocr_score
            if any(item[0] == "R4" for item in active_rules):
                level = "ALTO"
            elif risk < 0.33:
                level = "BAJO"
            elif risk < 0.66:
                level = "MEDIO"
            else:
                level = "ALTO"
            st.subheader("5. Resultado combinado")
            st.metric("Nivel de riesgo", level, f"Puntaje: {risk:.1%}")
            st.caption("Pesos y umbrales iniciales; deben calibrarse con datos de validación.")

    st.error(
        "Un resultado bajo no certifica que el medicamento sea original, eficaz o seguro. "
        "Esta herramienta es exclusivamente una evaluación preliminar del empaque."
    )

st.subheader("Fotografía del medicamento")
st.write("Toma una fotografía con la cámara del celular o selecciona una imagen.")

capture_mode = st.radio(
    "Método de captura",
    ["Tomar fotografía", "Subir fotografía"],
    horizontal=True,
    label_visibility="collapsed",
)

image_file = None
if capture_mode == "Tomar fotografía":
    st.caption("Permite el acceso a la cámara cuando el navegador lo solicite.")
    image_file = st.camera_input(
        "Enfoca el empaque y captura la fotografía",
        resolution="1080p",
        key="medicine_camera",
    )
else:
    image_file = st.file_uploader(
        "Selecciona una fotografía del empaque",
        type=["jpg", "jpeg", "png"],
        help="Formatos admitidos: JPG, JPEG y PNG.",
        key="medicine_upload",
    )

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        rgb = np.array(image)
        st.image(image, caption="Imagen seleccionada", use_container_width=True)
        analyze_image(rgb)
    except Exception as exc:
        st.error(f"No se pudo abrir la fotografía: {exc}")
else:
    st.info("Cuando captures o selecciones una imagen, el análisis aparecerá aquí.")

with st.expander("Limitaciones del prototipo"):
    st.write(
        "- Esta primera publicación valida la cámara y la calidad básica de la imagen.\n"
        "- Los módulos OCR y CNN se incorporarán después de preparar el dataset y entrenar los modelos.\n"
        "- La consulta del registro sanitario oficial de DIGEMID no está automatizada.\n"
        "- La herramienta no examina el contenido químico del medicamento."
    )

st.caption("FarmaCheck IA · Proyecto académico de Inteligencia Artificial Aplicada · Universidad de Lima")
