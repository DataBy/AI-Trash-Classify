import time
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import tensorflow as tf  # TM (TFLite)

# ====== Rutas a modelos ======
TACO_DET_WEIGHTS      = "modelo_02.pt"      # detector TACO
MATERIAL_YOLO_WEIGHTS = "modelo_01.pt"      # YOLO 5 clases
TM_TFLITE_PATH        = "modelo_03.tflite"  # Teachable Machine (TFLite)
TM_LABELS_PATH        = "labels.txt"

# ====== Dispositivo ======
USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else "cpu"

# ====== Umbrales y performance ======
DET_CONF_TH    = 0.45
MAT_CONF_TH    = 0.60
MAT_STRONG_TH  = 0.80
TM_CONF_TH     = 0.80
MIN_ROI_SIDE   = 40
BLUR_TH        = 60.0
DET_IMGSZ      = 640
MAT_IMGSZ      = 256
TM_IMGSZ       = 224
TM_NORM_NEG1_1 = True

# Inferir/accionar/loguear cada 5 s (video siempre en tiempo real)
INFER_INTERVAL_S = 5.0

WIN_NAME = "Clasificador de Residuos (TM + YOLOs)"

# ====== Mapeos ======
TACO_TO_BIN = {
    # Carton
    'Other carton': 'Carton', 'Egg carton': 'Carton', 'Drink carton': 'Carton',
    'Corrugated carton': 'Carton', 'Meal carton': 'Carton', 'Pizza box': 'Carton', 'Toilet tube': 'Carton',
    # Plastico
    'Other plastic bottle': 'Plastico', 'Clear plastic bottle': 'Plastico', 'Plastic bottle cap': 'Plastico',
    'Other plastic cup': 'Plastico', 'Disposable plastic cup': 'Plastico', 'Foam cup': 'Plastico',
    'Plastic lid': 'Plastico', 'Other plastic': 'Plastico', 'Plastic film': 'Plastico',
    'Other plastic wrapper': 'Plastico', 'Single-use carrier bag': 'Plastico', 'Polypropylene bag': 'Plastico',
    'Crisp packet': 'Plastico', 'Spread tub': 'Plastico', 'Tupperware': 'Plastico',
    'Disposable food container': 'Plastico', 'Foam food container': 'Plastico', 'Other plastic container': 'Plastico',
    'Plastic glooves': 'Plastico', 'Plastic utensils': 'Plastico', 'Plastic straw': 'Plastico', 'Six pack rings': 'Plastico',
    # Aluminio / Metal
    'Aluminium foil': 'Aluminio', 'Aluminium blister pack': 'Aluminio', 'Food Can': 'Aluminio',
    'Aerosol': 'Aluminio', 'Drink can': 'Aluminio', 'Metal bottle cap': 'Aluminio', 'Metal lid': 'Aluminio',
    'Pop tab': 'Aluminio', 'Scrap metal': 'Aluminio',
    # Papel
    'Normal paper': 'Papel', 'Magazine paper': 'Papel', 'Tissues': 'Papel', 'Wrapping paper': 'Papel',
    'Paper bag': 'Papel', 'Plastified paper bag': 'Papel', 'Paper straw': 'Papel', 'Paper cup': 'Papel'
}
MATERIAL_TO_BIN = {'Carton': 'Carton', 'Plastic': 'Plastico', 'Metal': 'Aluminio'}
TM_TO_BIN = {'aluminio': 'Aluminio', 'papel': 'Papel', 'plastico': 'Plastico', 'carton': 'Carton'}

CARTON_HINTS = {
    'Egg carton', 'Other carton', 'Drink carton', 'Corrugated carton',
    'Meal carton', 'Pizza box', 'Toilet tube', 'Paper cup'
}

# ====== Modelos YOLO ======
taco_det = YOLO(TACO_DET_WEIGHTS)
mat_yolo = YOLO(MATERIAL_YOLO_WEIGHTS)

# ====== Teachable Machine (TFLite) ======
with open(TM_LABELS_PATH, "r", encoding="utf-8") as f:
    TM_CLASSES = [ln.strip() for ln in f if ln.strip()]

tflite = tf.lite.Interpreter(model_path=TM_TFLITE_PATH)
tflite.allocate_tensors()
_TM_IN  = tflite.get_input_details()[0]['index']
_TM_OUT = tflite.get_output_details()[0]['index']

def tm_preprocess_tflite(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (TM_IMGSZ, TM_IMGSZ)).astype(np.float32)
    img = (img/127.5 - 1.0) if TM_NORM_NEG1_1 else (img/255.0)
    return np.expand_dims(img, 0)

def classify_with_tm(roi_bgr):
    if roi_bgr.size == 0:
        return "", 0.0
    x = tm_preprocess_tflite(roi_bgr)
    tflite.set_tensor(_TM_IN, x)
    tflite.invoke()
    probs = tflite.get_tensor(_TM_OUT)[0]
    idx = int(np.argmax(probs))
    return TM_CLASSES[idx], float(probs[idx])

# Warmup material
_dummy = np.zeros((MAT_IMGSZ, MAT_IMGSZ, 3), dtype=np.uint8)
mat_yolo(_dummy, imgsz=MAT_IMGSZ, device=DEVICE, verbose=False)

# ====== Utilidades ======
def clamp_bbox(x1, y1, x2, y2, w, h):
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w-1, int(x2)), min(h-1, int(y2))
    return x1, y1, x2, y2

def draw_box(frame, xyxy, color, txt):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, txt, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def draw_grid(frame, thirds=True, color=(200,200,200), thickness=1):
    h, w = frame.shape[:2]
    for i in (1, 2):
        x = w*i//3; y = h*i//3
        cv2.line(frame, (x,0), (x,h), color, thickness)
        cv2.line(frame, (0,y), (w,y), color, thickness)

def classify_material_with_yolo(roi_bgr):
    if roi_bgr.size == 0:
        return "", 0.0, "none"
    res = mat_yolo(roi_bgr, conf=0.25, iou=0.45, imgsz=MAT_IMGSZ, device=DEVICE, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return "", 0.0, "none"
    confs = res.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    box = res.boxes[best_idx]
    cls_idx = int(box.cls[0]); conf = float(box.conf[0])
    raw_label = res.names[cls_idx]
    mapped = MATERIAL_TO_BIN.get(raw_label, "")
    if mapped and conf >= MAT_CONF_TH:
        return mapped, conf, "CLF"
    return "", conf, "none"

def fuse_labels(taco_label, taco_conf, mat_label, mat_conf, tm_label=None, tm_conf=0.0):
    # 0) Prioridad TM
    if tm_label and tm_conf >= TM_CONF_TH:
        mapped_tm = TM_TO_BIN.get(tm_label.lower())
        if mapped_tm:
            return mapped_tm, tm_conf, "TM"
    # 1) Pistas fuertes de carton
    if taco_label in CARTON_HINTS and taco_conf >= DET_CONF_TH:
        if mat_label == "Carton" or mat_conf < MAT_STRONG_TH:
            return "Carton", max(taco_conf, mat_conf), "DET"
    # 2) Material
    if mat_label and mat_conf >= MAT_CONF_TH:
        return mat_label, mat_conf, "CLF"
    # 3) TACO (incluye Papel)
    mapped = TACO_TO_BIN.get(taco_label)
    if mapped and taco_conf >= DET_CONF_TH:
        return mapped, taco_conf, "DET"
    return "Incierto", 0.0, "none"

def center_crop_bbox(w, h, scale=0.6):
    side = int(min(w, h) * scale)
    x1 = (w - side)//2
    y1 = (h - side)//2
    x2 = x1 + side
    y2 = y1 + side
    return clamp_bbox(x1, y1, x2, y2, w, h)

# ====== Webcam (sin filtros de color) ======
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for prop, val in [(cv2.CAP_PROP_AUTO_WB, 1), (cv2.CAP_PROP_AUTOFOCUS, 1)]:
    try: cap.set(prop, val)
    except: pass
ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
if not ok:
    try: cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    except: pass

if not cap.isOpened():
    raise RuntimeError("No pude abrir la webcam.")

prev = time.time()
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280, 720)

show_grid = True

# Estado para throttling y overlays
next_infer_t   = 0.0                 # cuándo vuelve a inferir/accionar/loguear
last_boxes_out = []                  # [(x1,y1,x2,y2,label,conf,source)]
last_any       = False               # si hubo detecciones la ultima vez

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]

    # FPS visual
    now = time.time()
    fps = 1.0 / (now - prev) if now > prev else 0.0
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Nitidez (solo para overlay; no bloquea inferencia ya que inferimos por tiempo)
    sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    borroso = sharp < BLUR_TH
    if borroso:
        cv2.putText(frame, "Enfocando...", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

    # ====== Inferencia/Accion/Log CADA 5s ======
    do_infer = (now >= next_infer_t)
    if do_infer:
        # 1) Deteccion TACO sobre imagen reducida
        det_h = int(h * DET_IMGSZ / max(w, h))
        det_w = int(w * DET_IMGSZ / max(w, h))
        det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        sx, sy = w / det_w, h / det_h

        cached_boxes = []
        res = taco_det(det_frame, stream=False, conf=DET_CONF_TH, imgsz=DET_IMGSZ,
                       device=DEVICE, verbose=False)[0]
        names = res.names
        if res.boxes is not None:
            for b in res.boxes:
                det_conf = float(b.conf[0])
                x1d, y1d, x2d, y2d = b.xyxy[0].tolist()
                x1, y1, x2, y2 = x1d*sx, y1d*sy, x2d*sx, y2d*sy
                x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)
                if min(x2-x1, y2-y1) < MIN_ROI_SIDE:
                    continue
                cached_boxes.append((x1, y1, x2, y2, names[int(b.cls[0])], det_conf))

        last_boxes_out.clear()
        last_any = False

        # 2) Clasificacion por cajas detectadas
        for (x1, y1, x2, y2, taco_label, det_conf) in cached_boxes:
            last_any = True
            pad = 10
            rx1, ry1, rx2, ry2 = clamp_bbox(x1-pad, y1-pad, x2+pad, y2+pad, w, h)
            roi = frame[ry1:ry2, rx1:rx2].copy()
            if roi.size == 0:
                continue

            # Teachable Machine (PRIORIDAD)
            tm_label, tm_conf = classify_with_tm(roi)

            # YOLO material (secundario)
            mat_label, mat_conf, _ = classify_material_with_yolo(roi)

            # Fusion
            final_label, final_conf, source = fuse_labels(
                taco_label, det_conf, mat_label, mat_conf, tm_label, tm_conf
            )

            # Acciones y log cada 5s
            print(f"[{source}] {final_label} {final_conf:.2f} | "
                  f"taco={taco_label} {det_conf:.2f} "
                  f"mat={mat_label} {mat_conf:.2f} tm={tm_label} {tm_conf:.2f}")

            if not borroso:
                if final_label == "Plastico":  print("ACCION: Plastico -> motor A")
                elif final_label == "Carton":  print("ACCION: Carton -> motor B")
                elif final_label == "Aluminio":print("ACCION: Aluminio -> motor C")
                elif final_label == "Papel":   print("ACCION: Papel -> motor D")

            last_boxes_out.append((x1, y1, x2, y2, final_label, final_conf, source))

        # 3) Fallback TM centrado si no hubo detecciones
        if not last_any:
            cx1, cy1, cx2, cy2 = center_crop_bbox(w, h, scale=0.6)
            roi = frame[cy1:cy2, cx1:cx2].copy()
            tm_label, tm_conf = classify_with_tm(roi)
            mapped = TM_TO_BIN.get(tm_label.lower(), "")
            if mapped and tm_conf >= TM_CONF_TH:
                print(f"[TM] {mapped} {tm_conf:.2f} (fallback centro) tm={tm_label}")
                if not borroso:
                    if mapped == "Plastico":  print("ACCION: Plastico -> motor A")
                    elif mapped == "Carton":  print("ACCION: Carton -> motor B")
                    elif mapped == "Aluminio":print("ACCION: Aluminio -> motor C")
                    elif mapped == "Papel":   print("ACCION: Papel -> motor D")
                last_boxes_out.append((cx1, cy1, cx2, cy2, mapped, tm_conf, "TM"))

        # Programa la proxima inferencia
        next_infer_t = now + INFER_INTERVAL_S

    # ====== Dibujar ultimo resultado mientras el video sigue en tiempo real ======
    for (x1, y1, x2, y2, label, conf, source) in last_boxes_out:
        src_tag = {"CLF":"CLF", "DET":"DET", "TM":"TM", "none":"?"}.get(source, "?")
        color  = (0,255,0) if source=="CLF" else ((0,200,255) if source=="DET"
                    else ((255,255,0) if source=="TM" else (0,0,255)))
        draw_box(frame, (x1, y1, x2, y2), color, f"{label} {conf:.2f} [{src_tag}]")

    if show_grid:
        draw_grid(frame, thirds=True)

    cv2.imshow(WIN_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('g'):
        show_grid = not show_grid

cap.release()
cv2.destroyAllWindows()
