import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Metrics {
  ver: number; cv: number; pw: number; ipdr: number;
  exposure_tolerance_pct?: number; robust_pass_rate?: number; shape_error_um_p90?: number;
  dose_contrast: number; n_valid_iters: number;
  pct_layers_overlap: number; cv_z_p95: number; cv_z_worst: number;
  pw_z_p5: number; pw_z_worst: number; n_active_layers: number;
  elapsed_time: number; total_time: number; export_time: number;
  min_gel_dose: number; max_gel_dose: number; mean_gel_dose: number;
  max_void_dose: number; mean_void_dose: number;
  final_error: number; initial_error: number; error_improvement: number;
  converged_iter: number;
  iterations_run?: number | null;          // iteraciones reales del bucle CAL
  stop_reason?: "threshold" | "stall" | "max_iter" | "pw_margin" | "max_time" | null;
}
interface Orient { rx: number; ry: number; rz: number; reduction_pct?: number; }
interface VialInfo {
  vial: { outer_diameter_mm: number; wall_thickness_mm: number; n_vial: number; n_resin: number;
          n_outside: number; printable_diameter_mm: number; printable_radius_mm: number };
  max_delta_deg: number; min_transmission: number; active_columns?: number; mode?: "model" | "post";
}
interface JobSummary {
  score: number; score_bio: number; verdict: string;
  penalties: string[]; total_time: number; orient: Orient; metrics: Metrics;
  vial?: VialInfo | null;
  geometry?: { voxel_pitch_um: number; projector_pixel_um: number; part_radius_mm: number };
  robustness?: Robustness | null;
  score_breakdown?: ScoreBreakdown | null;
}
interface ScoreBreakdown {
  score: number; dose_score: number; tolerance_factor: number; robustness_factor: number | null;
  factor: number; weights: number[]; tol_ref_pct: number;
}
// vamcore.robustness: cuánto error real aguanta la pila de imágenes (cifras principales del resultado)
interface Robustness {
  exposure_tolerance_pct?: number; n_scenarios?: number; n_pass?: number; pass_rate?: number;
  shape_error_um_mean?: number; shape_error_um_p90?: number; shape_tol_um?: number;
  worst?: { shape_error_um: number; axis_offset_um: number; speed_err_pct: number; blur_um: number; dose_err_pct: number } | null;
  error?: string;
}
// Vial (corrección de refracción sin baño). material/resin = clave de preset o "custom"
interface VialCfg {
  enabled: boolean;
  outerD: number; wall: number;
  material: string; nVial: number;
  resin: string; nResin: number;
  mode?: "post" | "model"; // corrección posterior de las imágenes o refracción dentro del modelo (F6c)
}
interface VialPreview { printable_diameter_mm: number; printable_radius_mm: number; r_inner_mm: number; }
interface JobStatus {
  job_id: string; status: "running" | "done" | "error";
  progress: string; error?: string; summary?: JobSummary;
}
interface RPiStatus {
  status: "IDLE" | "CARGANDO_MODELO" | "PROYECTANDO" | "FINALIZADO" | "ERROR";
  progress_percent: number;
  current_frame: number;
  total_frames: number;
  elapsed_s: number;
}
interface LibraryItem {
  job_id: string; model_name: string; timestamp: string;
  score_bio: number; score: number; verdict: string;
  total_time: number; n_frames: number;
  orient: Orient; metrics: Metrics; penalties: string[];
  params: Record<string, any>;
  robustness?: Robustness | null;
  score_breakdown?: ScoreBreakdown | null;
  images: { reconstruction_slices?: string; layer_profile?: string; dose_histogram?: string; };
  all_frames: string[];
  frame_sample?: string;
  frames_base?: string;
  has_bitplanes?: boolean;
  n_bitplanes?: number;
  all_bitplanes?: string[];
  bitplanes_base?: string;
  bitplane_params?: {
    n_bitplanes:        number;
    projector_pixel_um: number;
    frame_w:            number;
    frame_h:            number;
    rotation_speed_dps: number;
    spi_ms_per_frame:   number;
  };
}
interface Config {
  numAngles: number;
  resolution: (typeof RESOLS)[number]; filter: string; method: string;
  stop: "ver" | "margin"; // criterio de parada: umbral VER o margen de ventana de proceso (MIGRACION.md F4/F5)
  pwMargin: number; // margen de ventana de proceso para parar con «Margen PW»
  absorptionCm: number; // α de la resina (cm⁻¹, Beer–Lambert, F6b); 0 = sin absorción
}
interface WifiNetwork {
  ssid: string;
  signal: number; // 0-100
  security: string;
  in_use: boolean;
}
interface WifiStatus {
  connected: boolean;
  ssid: string | null;
  ip: string | null;
  hostname: string;
  mdns: string;
  state: string;
}
interface CM4DiscoverResult {
  found: boolean;
  ip: string | null;
  hostname: string;
  cached: boolean;
  wifi?: WifiStatus;
  error?: string;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const API     = "http://localhost:8000";
const CYL_R   = 10;
const CYL_H   = 25;
const POLL_MS = 1200;
const FILTERS = ["shepp-logan", "ram-lak", "cosine", "hanning", "hamming"];
const METHODS = ["CAL"];
const RESOLS  = [64, 128, 256, 512] as const;
const DEF_CFG: Config = { numAngles: 360, resolution: 128, filter: "cosine", method: "CAL", stop: "ver", pwMargin: 0.03, absorptionCm: 0 };
const DEF_VIAL: VialCfg = { enabled: false, outerD: 20, wall: 1, material: "borosilicate", nVial: 1.483, resin: "gelma", nResin: 1.35, mode: "model" };
// Respaldo si el servidor no responde a /vial/presets (mismos valores que vamcore/optics, aprox. a 405 nm)
const FALLBACK_PRESETS = {
  vial_materials: { borosilicate: 1.483, fused_silica: 1.47, soda_lime: 1.53, pmma: 1.506, polystyrene: 1.62 } as Record<string, number>,
  resins: { water: 1.343, gelma: 1.35, acrylate: 1.48 } as Record<string, number>,
};
const PRESET_LABEL: Record<string, string> = {
  borosilicate: "Borosilicato", fused_silica: "Cuarzo fundido", soda_lime: "Vidrio común", pmma: "PMMA (acrílico)",
  polystyrene: "Poliestireno", water: "Agua", gelma: "GelMA", acrylate: "Acrilato", custom: "Otro (índice)",
};
// Panel widths — deben coincidir con el CSS del canvas fixed
const LEFT_W  = 300;
const RIGHT_W = 360;
// Proyector propio (MIGRACION.md F6): Joseph + retroproyección por vóxel, con la refracción del vial y la absorción de
// la resina dentro del modelo físico. Nombre en la interfaz:
const PROJECTOR_NAME = "Prisma";
const TOP_H   = 48;

const PROGRESS_STEPS = [
  { at: 0,  msg: "Iniciando motor de cálculo…" },
  { at: 10, msg: "Cargando geometría y tensores…" },
  { at: 25, msg: "Aplicando transformaciones manuales…" },
  { at: 35, msg: "Voxelizando malla a grill de 128³…" },
  { at: 50, msg: "Determinando factor d_h vía EDT…" },
  { at: 65, msg: "GPU: Optimizando sinograma MLEM…" },
  { at: 80, msg: "Finalizando y calculando métricas…" },
  { at: 90, msg: "Generando reporte visual de rodajas…" },
  { at: 95, msg: "Exportando secuencias para proyector…" },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

const fmt2  = (v?: number) => v != null && !isNaN(v) ? v.toFixed(2) : "—";
const fmt4  = (v?: number) => v != null && !isNaN(v) ? v.toFixed(4) : "—";
const fmts  = (v?: number) => v != null && !isNaN(v) ? `${v.toFixed(1)}s` : "—";
const fmtDt = (ts?: string) => !ts || ts.length < 13 ? (ts ?? "") :
  `${ts.slice(6,8)}/${ts.slice(4,6)}/${ts.slice(0,4)} ${ts.slice(9,11)}:${ts.slice(11,13)}`;

// El Score Bio evalúa la DOSIS SIMULADA de las imágenes generadas, no la impresión real.
const SCORE_NOTE = "Calidad de las imágenes generadas (dosis simulada). No garantiza el éxito de la impresión.";
const SCORE_TIP  = "Score Bio = nota de dosis simulada × (0,6 + 0,2·tolerancia de exposición respecto a ±5 % + 0,2·fracción de escenarios de robustez correctos). La nota de dosis mide cuán bien separa la dosis simulada la pieza del vacío (VER, PW, IPDR, CV). " +
  "La simulación no modela la absorción de la resina, la difusión, la inhibición por oxígeno, la respuesta no lineal " +
  "del material, el desenfoque del proyector ni errores mecánicos (eje de giro, velocidad, vibración). " +
  "Un 100/100 exige dosis perfecta, tolerancia ≥ ±5 % y todos los escenarios dentro; aun así, el resultado real debe validarse con impresiones.";
const VC: Record<string,string> = { EXCELENTE:"#059669",BUENO:"#2563eb",ACEPTABLE:"#d97706",POBRE:"#dc2626",FALLIDO:"#dc2626" };
const VB: Record<string,string> = { EXCELENTE:"#ecfdf5",BUENO:"#eff6ff",ACEPTABLE:"#fffbeb",POBRE:"#fef2f2",FALLIDO:"#fef2f2" };
const vc = (v?: string) => VC[v ?? ""] ?? "#94a3b8";
const vb = (v?: string) => VB[v ?? ""] ?? "#f8fafc";

function isOutside(mesh: THREE.Mesh): boolean {
  const box = new THREE.Box3().setFromObject(mesh);
  const c   = new THREE.Vector3(); box.getCenter(c);
  const s   = new THREE.Vector3(); box.getSize(s);
  return (
    Math.sqrt(c.x*c.x + c.z*c.z) + Math.max(s.x,s.z)/2 > CYL_R + 0.5 ||
    box.min.y < -0.5 || box.max.y > CYL_H + 0.5
  );
}

// ─── Three.js Hook ────────────────────────────────────────────────────────────
// El canvas vive en un div position:fixed que NUNCA se desmonta.
// Este hook inicializa la escena una sola vez, protegido por un flag en ref.

function useThreeScene(canvasRef: React.RefObject<HTMLCanvasElement | null>) {
  const R = useRef<{
    scene:    THREE.Scene | null;
    camera:   THREE.PerspectiveCamera | null;
    renderer: THREE.WebGLRenderer | null;
    orbit:    OrbitControls | null;
    tc:       TransformControls | null;
    mesh:     THREE.Mesh | null;
    meshMat:  THREE.MeshStandardMaterial | null;
    raf:      number;
    ready:    boolean;
  }>({
    scene: null, camera: null, renderer: null,
    orbit: null, tc: null, mesh: null, meshMat: null,
    raf: 0, ready: false,
  });

  const outsideCb = useRef<((v: boolean) => void) | undefined>(undefined);
  const prevOut   = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const r = R.current;

    // Inicialización única (si no se ha hecho o si el renderer se perdió)
    if (!r.ready) {
      r.ready = true;
      
      // -- Renderer --
      r.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      r.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      r.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
      r.renderer.shadowMap.enabled = true;

      // -- Scene --
      r.scene = new THREE.Scene();
      r.scene.background = new THREE.Color(0xf8fafc);

      // -- Camera --
      r.camera = new THREE.PerspectiveCamera(42, canvas.clientWidth / canvas.clientHeight, 0.1, 2000);
      r.camera.position.set(90, 70, 90);

      // -- OrbitControls --
      r.orbit = new OrbitControls(r.camera, canvas);
      r.orbit.enableDamping = true;
      r.orbit.dampingFactor = 0.08;
      r.orbit.minDistance = 15;
      r.orbit.maxDistance = 500;
      r.orbit.target.set(0, CYL_H / 2, 0);
      r.orbit.update();

      // -- Scene Content: Lights --
      r.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
      const sun = new THREE.DirectionalLight(0xffffff, 1.0);
      sun.position.set(80, 120, 80); sun.castShadow = true; r.scene.add(sun);
      const fill = new THREE.DirectionalLight(0xdbeafe, 0.4);
      fill.position.set(-60, 40, -60); r.scene.add(fill);

      // -- Scene Content: Workspace Cylinder --
      const bodyMat = new THREE.MeshPhysicalMaterial({
        color: 0x3b82f6, transparent: true, opacity: 0.05,
        side: THREE.DoubleSide, depthWrite: false,
      });
      const body = new THREE.Mesh(new THREE.CylinderGeometry(CYL_R, CYL_R, CYL_H, 128, 1, true), bodyMat);
      body.position.y = CYL_H / 2;
      r.scene.add(body);

      const ringMat = new THREE.MeshBasicMaterial({ color:0x3b82f6, transparent:true, opacity:0.55, side:THREE.DoubleSide });
      const discMat = new THREE.MeshBasicMaterial({ color:0x3b82f6, transparent:true, opacity:0.10, side:THREE.DoubleSide, depthWrite:false });
      for (const y of [0.01, CYL_H-0.01]) {
        const disc = new THREE.Mesh(new THREE.CircleGeometry(CYL_R, 128), discMat.clone());
        disc.rotation.x = -Math.PI/2; disc.position.y = y; r.scene.add(disc);
        const ring = new THREE.Mesh(new THREE.RingGeometry(CYL_R-0.5, CYL_R, 128), ringMat.clone());
        ring.rotation.x = -Math.PI/2; ring.position.y = y+0.01; r.scene.add(ring);
      }

      // -- Scene Content: Grid --
      const grid = new THREE.GridHelper(160, 32, 0xdde4ee, 0xdde4ee);
      (grid.material as any).transparent = true;
      (grid.material as any).opacity = 0.7;
      grid.position.y = -0.1;
      r.scene.add(grid);

      // -- TransformControls --
      try {
        const tc = new TransformControls(r.camera, canvas);
        tc.addEventListener("dragging-changed", (e: any) => { if (r.orbit) r.orbit.enabled = !e.value; });
        tc.addEventListener("objectChange", () => {
          if (!r.mesh) return;
          r.mesh.position.set(0, CYL_H / 2, 0); // Magnetizado al centro del vial
          window.dispatchEvent(new CustomEvent("vam-dirty"));
          const out = isOutside(r.mesh);
          if (out !== prevOut.current) {
            prevOut.current = out;
            if (r.meshMat) r.meshMat.color.set(out ? 0xef4444 : 0x475569);
            outsideCb.current?.(out);
          }
        });
        if (tc instanceof THREE.Object3D) r.scene.add(tc); 
        else if ((tc as any).getHelper?.()) r.scene.add((tc as any).getHelper());
        r.tc = tc;
      } catch(e) { console.warn("TC fail", e); }
    }

    // -- Lifecycle: Resize --
    const ro = new ResizeObserver(() => {
      if (!r.renderer || !r.camera) return;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (w === 0 || h === 0) return;
      r.renderer.setSize(w, h, false);
      r.camera.aspect = w / h;
      r.camera.updateProjectionMatrix();
    });
    ro.observe(canvas);

    // -- Lifecycle: Animation Loop --
    const tick = () => {
      r.raf = requestAnimationFrame(tick);
      if (r.orbit) r.orbit.update();
      if (r.renderer && r.scene && r.camera) {
        r.renderer.render(r.scene, r.camera);
      }
    };
    tick();

    return () => {
      cancelAnimationFrame(r.raf);
      ro.disconnect();
    };
  }, []);

  // -- API --

  const loadSTL = useCallback((buffer: ArrayBuffer) => {
    const r = R.current;
    if (!r.scene || !r.camera || !r.renderer || !r.orbit) return;

    if (r.mesh) {
      r.tc?.detach();
      r.scene.remove(r.mesh);
      r.mesh.geometry.dispose();
    }

    const geo = new STLLoader().parse(buffer);
    geo.computeBoundingBox();
    geo.computeVertexNormals();
    geo.center(); 
    // Corrección visual: Alinear STL (Z-up) con Three.js (Y-up)
    geo.applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
    const bb = geo.boundingBox!;
    const sz = new THREE.Vector3(); bb.getSize(sz);

    r.meshMat = new THREE.MeshStandardMaterial({
      color: 0x475569, metalness: 0.1, roughness: 0.5,
      transparent: true, opacity: 0.9,
    });
    r.mesh = new THREE.Mesh(geo, r.meshMat);
    r.mesh.castShadow = true;
    r.mesh.receiveShadow = true;
    r.mesh.position.set(0, CYL_H / 2, 0); // Magnetizado al centro
    r.scene.add(r.mesh);
    prevOut.current = false;

    // Enfocar cámara
    const d = Math.max(sz.x, sz.y, sz.z, 25);
    // Auto-zoom desactivado a petición del usuario
    // r.camera.position.set(d * 1.5, CYL_H / 2 + d * 0.8, d * 1.5);
    r.orbit.target.set(0, CYL_H / 2, 0); // Siempre al centro del vial
    r.orbit.update();
    r.renderer.render(r.scene, r.camera);
    window.dispatchEvent(new CustomEvent("vam-dirty")); // avisa a los paneles (radio de la pieza, etc.)
  }, []);

  const applyOrientation = useCallback((rx: number, ry: number, rz: number = 0) => {
    const r = R.current;
    if (!r.mesh) return;
    r.mesh.rotation.set(THREE.MathUtils.degToRad(rx), THREE.MathUtils.degToRad(ry), THREE.MathUtils.degToRad(rz));
    r.mesh.position.set(0, CYL_H / 2, 0);
    window.dispatchEvent(new CustomEvent("vam-dirty"));
  }, []);

  const setTransformMode = useCallback((mode: "rotate" | null) => {
    const r = R.current;
    if (!r.tc) return;
    if (mode === null) {
      r.tc.detach();
    } else if (r.mesh) {
      r.tc.attach(r.mesh);
      r.tc.setMode(mode);
    }
  }, []);

  const resetTransform = useCallback(() => {
    const r = R.current;
    if (!r.mesh) return;
    r.tc?.detach();
    r.mesh.position.set(0, CYL_H / 2, 0);
    r.mesh.rotation.set(0, 0, 0);
    r.mesh.scale.set(1, 1, 1);
    if (r.meshMat) r.meshMat.color.set(0x475569);
    prevOut.current = false;
    outsideCb.current?.(false);
    window.dispatchEvent(new CustomEvent("vam-dirty"));
  }, []);

  const getTransform = useCallback(() => {
    const r = R.current;
    if (!r.mesh) return { rx: 0, ry: 0, rz: 0 };
    return {
      rx: THREE.MathUtils.radToDeg(r.mesh.rotation.x),
      ry: THREE.MathUtils.radToDeg(r.mesh.rotation.y),
      rz: THREE.MathUtils.radToDeg(r.mesh.rotation.z),
    };
  }, []);

  // ── Sistema de impresión ↔ visor ──────────────────────────────────────────
  // El visor es Y-arriba y el STL se carga girado −90° en X (C). cal_core y vamcore.orient trabajan en el sistema
  // del STL (Z = eje de giro) con R = Rz(rz)·Ry(ry)·Rx(rx) (Euler 'ZYX' de three.js). Antes se mandaban los ángulos
  // del visor tal cual: solo X coincidía; girar en Y del visor (eje vertical) inclinaba la pieza en la impresión.
  const C_ = useRef(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
  const Cinv_ = useRef(new THREE.Matrix4().makeRotationX(Math.PI / 2));

  const getPrintRotation = useCallback(() => {
    const r = R.current;
    if (!r.mesh) return { rx: 0, ry: 0, rz: 0 };
    const Mv = new THREE.Matrix4().makeRotationFromQuaternion(r.mesh.quaternion);
    const Ms = new THREE.Matrix4().multiplyMatrices(Cinv_.current, Mv).multiply(C_.current);
    const e = new THREE.Euler().setFromRotationMatrix(Ms, "ZYX");
    const d = (a: number) => { const v = THREE.MathUtils.radToDeg(a); return Math.abs(v) < 1e-9 ? 0 : v; };
    return { rx: d(e.x), ry: d(e.y), rz: d(e.z) };
  }, []);

  const applyPrintMatrix = useCallback((Ms: THREE.Matrix4) => {
    const r = R.current;
    if (!r.mesh) return;
    const Mv = new THREE.Matrix4().multiplyMatrices(C_.current, Ms).multiply(Cinv_.current);
    r.mesh.quaternion.setFromRotationMatrix(Mv);
    r.mesh.position.set(0, CYL_H / 2, 0);
    window.dispatchEvent(new CustomEvent("vam-dirty"));
  }, []);

  // Ángulos en el sistema de impresión (los que ve cal_core)
  const applyPrintRotation = useCallback((rx: number, ry: number, rz: number = 0) => {
    const e = new THREE.Euler(THREE.MathUtils.degToRad(rx), THREE.MathUtils.degToRad(ry), THREE.MathUtils.degToRad(rz), "ZYX");
    applyPrintMatrix(new THREE.Matrix4().makeRotationFromEuler(e));
  }, [applyPrintMatrix]);

  // Matriz 3×3 (filas) en el sistema de impresión, p. ej. la que devuelve POST /orient
  const applyPrintRotationMatrix = useCallback((m: number[][]) => {
    const Ms = new THREE.Matrix4().set(
      m[0][0], m[0][1], m[0][2], 0,
      m[1][0], m[1][1], m[1][2], 0,
      m[2][0], m[2][1], m[2][2], 0,
      0, 0, 0, 1);
    applyPrintMatrix(Ms);
  }, [applyPrintMatrix]);

  const setSnap = useCallback((snap: boolean) => {
    if (R.current.tc) R.current.tc.setRotationSnap(snap ? THREE.MathUtils.degToRad(5) : null);
  }, []);

  const setOutsideCb = useCallback((cb: (v: boolean) => void) => {
    outsideCb.current = cb;
  }, []);

  // Radio máximo (mm) de la pieza respecto al eje de giro. cal_core centra la caja envolvente de la
  // pieza ya orientada en el eje, así que se mide desde el centro de esa caja en el plano horizontal (XZ).
  const getPartRadius = useCallback((): number | null => {
    const r = R.current;
    if (!r.mesh) return null;
    r.mesh.updateMatrixWorld(true);
    const pos = r.mesh.geometry.getAttribute("position");
    const v = new THREE.Vector3();
    const box = new THREE.Box3().setFromObject(r.mesh);
    const cx = (box.min.x + box.max.x) / 2, cz = (box.min.z + box.max.z) / 2;
    let rmax = 0;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(r.mesh.matrixWorld);
      rmax = Math.max(rmax, Math.hypot(v.x - cx, v.z - cz));
    }
    return rmax;
  }, []);

  return { loadSTL, applyOrientation, setTransformMode, resetTransform, getTransform, setOutsideCb, setSnap, getPartRadius,
           getPrintRotation, applyPrintRotation, applyPrintRotationMatrix };
}

// ─── WiFi Panel ──────────────────────────────────────────────────────────────

function SignalBars({ signal }: { signal: number }) {
  const bars = 4;
  const filled = signal >= 80 ? 4 : signal >= 60 ? 3 : signal >= 35 ? 2 : 1;
  const color  = signal >= 60 ? "#059669" : signal >= 35 ? "#d97706" : "#ef4444";
  return (
    <span style={{ display:"inline-flex", alignItems:"flex-end", gap:1.5, height:12, marginLeft:4 }}>
      {Array.from({length: bars}, (_, i) => (
        <span key={i} style={{
          width: 3, borderRadius: 1,
          height: 3 + i * 2.5,
          background: i < filled ? color : "#e2e8f0",
        }} />
      ))}
    </span>
  );
}



// ─── Sub-components ───────────────────────────────────────────────────────────

function Row({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", padding:"5px 0", borderBottom:"1px solid #f1f5f9" }}>
      <span style={{ fontSize:11, color:"#94a3b8", textTransform:"uppercase", letterSpacing:"0.07em" }}>{label}</span>
      <span style={{ fontSize:12, fontFamily:"'JetBrains Mono',monospace", color:accent?"#059669":"#334155", fontWeight:accent?600:400 }}>{value}</span>
    </div>
  );
}

// Tarjeta y cifra compactas (paneles laterales)
function Card({ title, right, children, tip, tone }: { title: string; right?: React.ReactNode; children: React.ReactNode; tip?: string; tone?: { bg: string; border: string } }) {
  return (
    <div title={tip} style={{ background:tone?.bg ?? "#fff", border:`1px solid ${tone?.border ?? "#e2e8f0"}`, borderRadius:10, padding:"8px 10px", cursor:tip ? "help" : undefined }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:6 }}>
        <span style={{ fontSize:10, color:"#94a3b8", textTransform:"uppercase", letterSpacing:"0.1em", fontWeight:600 }}>{title}</span>
        {right}
      </div>
      {children}
    </div>
  );
}
function Stat({ label, value, color, big }: { label: string; value: string; color?: string; big?: boolean }) {
  return (
    <div title={label} style={{ background:"#f8fafc", borderRadius:7, padding:"4px 7px", minWidth:0 }}>
      <div style={{ fontSize:8.5, color:"#94a3b8", textTransform:"uppercase", letterSpacing:"0.05em", whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{label}</div>
      <div style={{ fontSize:big ? 16 : 12, fontWeight:big ? 700 : 500, fontFamily:"'JetBrains Mono',monospace", color:color ?? "#334155", marginTop:1, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{value}</div>
    </div>
  );
}
function StatGrid({ cols = 2, children }: { cols?: number; children: React.ReactNode }) {
  return <div style={{ display:"grid", gridTemplateColumns:`repeat(${cols}, minmax(0,1fr))`, gap:5 }}>{children}</div>;
}

// Fiabilidad estimada: tolerancia de exposición + prueba de robustez. Va debajo del Score Bio.
const ROBUST_TIP = "Tolerancia de exposición: cuánto puede desviarse la dosis real (potencia del LED o tiempo) sin que " +
  "la pieza falle, con el umbral de la resina centrado. Robustez: la pila se reconstruye con perturbaciones " +
  "(eje de giro, velocidad y ángulo de la plataforma, desenfoque, parpadeo del LED, error de dosis) y se mide el " +
  "error de forma. Las magnitudes de las perturbaciones son hipótesis por defecto, no medidas de tu máquina. " +
  "No modela absorción de la resina, difusión ni inhibición por oxígeno.";
function RobustBox({ tolPct, rob }: { tolPct?: number; rob?: Robustness | null }) {
  const tol = tolPct ?? rob?.exposure_tolerance_pct;
  if (tol == null && !rob) return null;
  const rate = rob?.pass_rate;
  const col = rate == null ? "#475569" : rate >= 0.9 ? "#059669" : rate >= 0.6 ? "#d97706" : "#dc2626";
  const tcol = tol == null ? "#475569" : tol >= 3 ? "#059669" : tol >= 1.5 ? "#d97706" : "#dc2626";
  return (
    <Card title="Fiabilidad estimada" tip={ROBUST_TIP} right={<span style={{ fontSize:9, color:"#cbd5e1" }}>hipótesis · validar</span>}>
      <StatGrid cols={3}>
        <Stat label="Exposición" value={tol != null ? `±${tol.toFixed(1)}%` : "—"} color={tcol} big />
        <Stat label="Escenarios" value={rob?.n_pass != null ? `${rob.n_pass}/${rob.n_scenarios}` : "—"} color={col} big />
        <Stat label="Forma p90" value={rob?.shape_error_um_p90 != null ? `${rob.shape_error_um_p90.toFixed(0)}µm` : "—"} big />
      </StatGrid>
      {rob?.worst && (
        <div style={{ marginTop:5, fontSize:9.5, color:"#94a3b8", whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>
          Peor {rob.worst.shape_error_um.toFixed(0)} µm (tol. {rob.shape_tol_um?.toFixed(0) ?? "—"}): eje {rob.worst.axis_offset_um.toFixed(0)} µm · vel. {rob.worst.speed_err_pct.toFixed(1)}% · dosis {rob.worst.dose_err_pct.toFixed(1)}%
        </div>
      )}
      {rob?.error && <div style={{ marginTop:6, fontSize:10, color:"#dc2626" }}>Robustez no disponible: {rob.error}</div>}
    </Card>
  );
}

// Score Bio: nota final /100 = dosis simulada × fiabilidad (tolerancia de exposición y robustez)
const fmtScore = (v?: number | null) => v == null ? "—" : String(Math.round(v));
function ScoreBox({ verdict, score, bd, tolPct, rob, penalties }: {
  verdict: string; score?: number | null; globalScore?: number | null; bd?: ScoreBreakdown | null;
  tolPct?: number; rob?: Robustness | null; big?: number; penalties?: string[];
}) {
  const tol = tolPct ?? rob?.exposure_tolerance_pct;
  return (
    <Card title="Score Bio · simulado" tip={SCORE_TIP} tone={{ bg:vb(verdict), border:`${vc(verdict)}33` }}
          right={<span style={{ fontSize:10, fontWeight:700, color:vc(verdict), letterSpacing:"0.08em" }}>{verdict}</span>}>
      <div style={{ display:"flex", alignItems:"baseline", gap:4, color:vc(verdict), lineHeight:1 }}>
        <span style={{ fontSize:34, fontWeight:700 }}>{fmtScore(score)}</span>
        <span style={{ fontSize:15, fontWeight:600, opacity:0.7 }}>/100</span>
        {bd && (
          <span style={{ marginLeft:"auto", fontSize:10.5, color:"#475569", textAlign:"right", lineHeight:1.5 }}>
            dosis {fmtScore(bd.dose_score)} · exp. {tol != null ? `±${tol.toFixed(1)}%` : "—"}<br/>
            robustez {rob?.n_pass != null ? `${rob.n_pass}/${rob.n_scenarios}` : "—"}
          </span>
        )}
      </div>
      {(penalties?.length ?? 0) > 0 && (
        <div title={penalties!.join("\n")} style={{ marginTop:5, fontSize:9.5, color:"#92400e", padding:"2px 6px", background:"#fffbeb", borderRadius:4, borderLeft:"2px solid #fbbf24", whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>
          {penalties!.join(" · ")}
        </div>
      )}
      <div style={{ marginTop:5, fontSize:9, color:"#64748b", lineHeight:1.3 }}>{SCORE_NOTE}</div>
    </Card>
  );
}

const STOP_LABEL: Record<string, string> = { threshold:"objetivo", stall:"estancado", max_iter:"máximo", pw_margin:"margen PW", max_time:"tiempo" };
function DoseCard({ m }: { m?: Metrics }) {
  return (
    <Card title="Dosis simulada" tip="VER: error volumétrico · PW: ventana de proceso (negativa = hay ventana) · CV: uniformidad en la pieza · IPDR: rango de dosis en la pieza · por capa: peor 5 %/95 % de las capas">
      <StatGrid cols={4}>
        <Stat label="VER" value={fmt4(m?.ver)} />
        <Stat label="PW" value={fmt4(m?.pw)} color={m?.pw != null && m.pw < 0 ? "#059669" : undefined} />
        <Stat label="CV" value={fmt4(m?.cv)} />
        <Stat label="IPDR" value={fmt4(m?.ipdr)} />
        <Stat label="Contraste" value={fmt2(m?.dose_contrast)} />
        <Stat label="CV z p95" value={fmt4(m?.cv_z_p95)} />
        <Stat label="PW z p5" value={fmt4(m?.pw_z_p5)} />
        <Stat label="Solape" value={`${fmt2(m?.pct_layers_overlap)}%`} />
      </StatGrid>
    </Card>
  );
}
function VialCard({ v }: { v?: VialInfo | null }) {
  if (!v) return null;
  return (
    <Card title="Vial sin baño" right={<span style={{ fontSize:9.5, color:"#94a3b8" }}>{v.mode === "post" ? "corrección posterior" : "refracción en el modelo"}</span>}>
      <StatGrid cols={4}>
        <Stat label="Zona impr." value={`Ø${v.vial.printable_diameter_mm.toFixed(1)}`} />
        <Stat label="Desv. máx" value={`${v.max_delta_deg.toFixed(1)}°`} />
        <Stat label="Transm. mín" value={`${(100 * v.min_transmission).toFixed(0)}%`} />
        <Stat label="n vial/resina" value={`${v.vial.n_vial.toFixed(2)}/${v.vial.n_resin.toFixed(2)}`} />
      </StatGrid>
    </Card>
  );
}
function ProcessCard({ m, orient, totalTime, frames, projector }: { m?: Metrics; orient?: Orient; totalTime?: number; frames?: number; projector?: string }) {
  const it = m?.iterations_run != null ? `${m.iterations_run}` : String(m?.n_valid_iters ?? "—");
  return (
    <Card title="Proceso" right={projector ? <span style={{ fontSize:9.5, color:"#94a3b8" }}>proyector {projector}</span> : undefined}>
      <StatGrid cols={4}>
        <Stat label={`Iter.${m?.stop_reason ? " · " + (STOP_LABEL[m.stop_reason] ?? m.stop_reason) : ""}`} value={it} />
        <Stat label="Optimiz." value={fmts(m?.elapsed_time)} />
        <Stat label="Total" value={fmts(totalTime ?? m?.total_time)} color="#059669" />
        <Stat label="Rx·Ry·Rz" value={`${orient?.rx?.toFixed(0) ?? "0"}·${orient?.ry?.toFixed(0) ?? "0"}·${orient?.rz?.toFixed(0) ?? "0"}°`} />
      </StatGrid>
      {frames != null && <div style={{ marginTop:4, fontSize:9.5, color:"#94a3b8" }}>{frames} fotogramas para el proyector</div>}
    </Card>
  );
}

function Sec({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontSize:10, color:"#cbd5e1", textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:6, fontWeight:600 }}>{title}</div>
      {children}
    </div>
  );
}

// ─── Progress Button ──────────────────────────────────────────────────────────

function ProgressButton({ startTime, totalSec, onStep }: { startTime: number; totalSec: number; onStep?: (msg: string) => void }) {
  const [pct, setPct] = useState(0);
  const [timeLeft, setTimeLeft] = useState("");

  useEffect(() => {
    const id = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      const rawPct = (elapsed / totalSec) * 100;
      const final = Math.min(rawPct, 99);
      setPct(final);

      const rem = Math.max(0, totalSec - elapsed);
      const m = Math.floor(rem / 60);
      const s = Math.floor(rem % 60).toString().padStart(2, "0");
      setTimeLeft(m === 0 && s === "00" ? "Finalizando..." : `${m}:${s} aprox.`);

      const step = [...PROGRESS_STEPS].reverse().find(s => final >= s.at);
      if (step && onStep) onStep(step.msg);
    }, 150);
    return () => clearInterval(id);
  }, [startTime, onStep]);

  return (
    <div style={{ position:"relative", height:52, borderRadius:10, overflow:"hidden", border:"1px solid #fee2e2", background:"#fef2f2" }}>
      <div style={{ position:"absolute", inset:0, width:`${pct}%`, background:"#ef4444", transition:"width 0.2s linear" }} />
      <div style={{ position:"absolute", inset:0, background:"linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent)", animation:"shimmer 1s infinite" }} />
      <div style={{ position:"absolute", inset:0, display:"flex", alignItems:"center", justifyContent:"center" }}>
        <span style={{ fontSize:18, fontWeight:800, color:"#fff", fontFamily:"'JetBrains Mono',monospace", textShadow:"0 1px 4px rgba(0,0,0,0.2)" }}>{timeLeft}</span>
      </div>
    </div>
  );
}

// ─── Config Panel ─────────────────────────────────────────────────────────────

// Controles compactos del panel izquierdo
const fieldRow: React.CSSProperties = { display:"flex", justifyContent:"space-between", alignItems:"center", gap:8, padding:"5px 0" };
const fieldLbl: React.CSSProperties = { fontSize:11, color:"#64748b" };
const inputS: React.CSSProperties = { width:68, fontSize:12, fontFamily:"'JetBrains Mono',monospace", border:"1px solid #e2e8f0", borderRadius:5, padding:"3px 7px", background:"#f8fafc", color:"#334155", textAlign:"right", outline:"none" };
function Chips<T extends string | number>({ options, value, onChange, labels }: { options: readonly T[]; value: T; onChange: (v: T) => void; labels?: Record<string, string> }) {
  return (
    <div style={{ display:"flex", gap:3, flexWrap:"wrap", justifyContent:"flex-end" }}>
      {options.map(o => {
        const on = o === value;
        return (
          <button key={String(o)} onClick={() => onChange(o)} style={{ padding:"3px 8px", border:`1px solid ${on?"#2563eb":"#e2e8f0"}`, borderRadius:5, background:on?"#eff6ff":"#f8fafc", color:on?"#2563eb":"#64748b", fontSize:11, fontWeight:on?700:400, cursor:"pointer", fontFamily:"inherit" }}>
            {labels?.[String(o)] ?? String(o)}
          </button>
        );
      })}
    </div>
  );
}

// ─── Optimización (+ Avanzado) ────────────────────────────────────────────────

function ConfigPanel({ cfg, onChange, useCuda, setUseCuda }: { cfg: Config; onChange: (c: Config) => void; useCuda: boolean; setUseCuda: (v: boolean) => void }) {
  const [adv, setAdv] = useState(false);
  const set = (k: keyof Config, v: unknown) => {
    onChange({ ...cfg, [k]: v });
    window.dispatchEvent(new CustomEvent("vam-dirty"));
  };
  const stop = cfg.stop ?? "ver";
  return (
    <Card title="Optimización">
      <div style={fieldRow}>
        <span style={fieldLbl}>Resolución</span>
        <Chips options={RESOLS} value={cfg.resolution} onChange={r => set("resolution", r)} />
      </div>
      <div style={fieldRow}>
        <span style={fieldLbl}>Ángulos</span>
        <input type="number" min={36} max={3600} step={1} value={cfg.numAngles} style={inputS}
          onChange={e => { const v=parseInt(e.target.value); if(!isNaN(v)&&v>=36&&v<=3600) set("numAngles",v); }} />
      </div>
      <div style={fieldRow} title="Umbral VER: para cuando el error volumétrico baja del umbral (lo habitual). Margen PW: para cuando hay una ventana de proceso de al menos ese margen; mucho más rápido, a cambio de menos tolerancia de exposición">
        <span style={fieldLbl}>Parada</span>
        <Chips options={["ver", "margin"] as const} value={stop} labels={{ ver:"Umbral VER", margin:"Margen PW" }}
          onChange={v => set("stop", v)} />
      </div>
      {stop === "margin" && (
        <div style={fieldRow} title="Ventana de proceso mínima (dosis normalizada a 1)">
          <span style={fieldLbl}>Margen PW</span>
          <input type="number" min={0.005} max={0.2} step={0.005} value={cfg.pwMargin ?? 0.03} style={inputS}
            onChange={e => { const v=parseFloat(e.target.value); if(!isNaN(v)&&v>0&&v<0.5) set("pwMargin",v); }} />
        </div>
      )}
      <button onClick={() => setAdv(a => !a)} style={{ marginTop:4, padding:"4px 0", border:"none", background:"transparent", color:"#94a3b8", fontSize:10.5, cursor:"pointer", fontFamily:"inherit", textAlign:"left" }}>
        {adv ? "▾" : "▸"} Avanzado
      </button>
      {adv && (
        <div style={{ borderTop:"1px solid #f1f5f9", paddingTop:4 }}>
          <div style={fieldRow}>
            <span style={fieldLbl}>Filtro FBP</span>
            <select value={cfg.filter} onChange={e => set("filter", e.target.value)} style={{ ...selS, maxWidth:120 }}>
              {FILTERS.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
          <div style={fieldRow}>
            <span style={fieldLbl}>Cálculo</span>
            <Chips options={["gpu", "cpu"] as const} value={useCuda ? "gpu" : "cpu"} labels={{ gpu:"GPU · CUDA", cpu:"CPU" }}
              onChange={v => setUseCuda(v === "gpu")} />
          </div>
        </div>
      )}
    </Card>
  );
}

// ─── Vial Panel ───────────────────────────────────────────────────────────────
// Corrección de la refracción del vial para imprimir sin baño de index matching.

const lbl: React.CSSProperties = { fontSize:11, color:"#94a3b8", textTransform:"uppercase", letterSpacing:"0.07em" };
const rowS: React.CSSProperties = { display:"flex", justifyContent:"space-between", alignItems:"center", padding:"6px 0", borderBottom:"1px solid #f1f5f9", gap:8 };
const numS: React.CSSProperties = { width:58, fontSize:12, fontFamily:"'JetBrains Mono',monospace", border:"1px solid #e2e8f0", borderRadius:5, padding:"3px 7px", background:"#f8fafc", color:"#334155", textAlign:"right", outline:"none" };
const selS: React.CSSProperties = { fontSize:11, border:"1px solid #e2e8f0", borderRadius:5, padding:"3px 7px", background:"#f8fafc", color:"#334155", fontFamily:"inherit", cursor:"pointer", outline:"none", maxWidth:130 };

function NumField({ value, min, max, step, onChange }: { value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  // Estado local para poder escribir libremente; solo se propaga un valor válido
  const [txt, setTxt] = useState(String(value));
  useEffect(() => { setTxt(String(value)); }, [value]);
  return (
    <input type="number" min={min} max={max} step={step} value={txt} style={numS}
      onChange={e => { setTxt(e.target.value); const v = parseFloat(e.target.value); if (!isNaN(v) && v >= min && v <= max) onChange(v); }} />
  );
}

function VialPanel({ vial, onChange, presets, preview, previewErr, partRadius, absorption, onAbsorption }: {
  vial: VialCfg; onChange: (v: VialCfg) => void;
  presets: typeof FALLBACK_PRESETS;
  preview: VialPreview | null; previewErr: string | null; partRadius: number | null;
  absorption: number; onAbsorption: (v: number) => void;
}) {
  const set = (patch: Partial<VialCfg>) => { onChange({ ...vial, ...patch }); window.dispatchEvent(new CustomEvent("vam-dirty")); };
  const fits = preview && partRadius != null ? partRadius <= preview.printable_radius_mm : null;
  return (
    <Card title="Vial y resina">
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:5, marginBottom:4 }}>
        {[{ label:"Con baño", val:false }, { label:"Vial sin baño", val:true }].map(({ label, val }) => (
          <button key={label} onClick={() => set({ enabled: val })} title={val ? "Sin baño de index matching: se modela la refracción del vial" : "El vial está en un baño de index matching: sin refracción"}
            style={{ padding:"5px 0", border:`1.5px solid ${vial.enabled===val?"#2563eb":"#e2e8f0"}`, borderRadius:7, background:vial.enabled===val?"#eff6ff":"#f8fafc", color:vial.enabled===val?"#2563eb":"#94a3b8", fontSize:11, fontWeight:vial.enabled===val?600:400, fontFamily:"inherit", cursor:"pointer" }}>{label}</button>
        ))}
      </div>
      {vial.enabled && (
        <>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"0 10px" }}>
            <div style={fieldRow} title="Diámetro exterior del vial (mm)"><span style={fieldLbl}>Ø ext.</span><NumField value={vial.outerD} min={1} max={100} step={0.1} onChange={v => set({ outerD: v })} /></div>
            <div style={fieldRow} title="Grosor de la pared del vial (mm)"><span style={fieldLbl}>Pared</span><NumField value={vial.wall} min={0} max={20} step={0.05} onChange={v => set({ wall: v })} /></div>
          </div>
          <div style={fieldRow}>
            <span style={fieldLbl}>Material</span>
            <select value={vial.material} style={selS}
              onChange={e => { const m = e.target.value; set(m === "custom" ? { material: m } : { material: m, nVial: presets.vial_materials[m] }); }}>
              {Object.keys(presets.vial_materials).map(k => <option key={k} value={k}>{PRESET_LABEL[k] ?? k} · {presets.vial_materials[k]}</option>)}
              <option value="custom">{PRESET_LABEL.custom}</option>
            </select>
          </div>
          {vial.material === "custom" && <div style={fieldRow}><span style={fieldLbl}>n vial</span><NumField value={vial.nVial} min={1} max={3} step={0.001} onChange={v => set({ nVial: v })} /></div>}
        </>
      )}
      <div style={fieldRow}>
        <span style={fieldLbl}>Resina</span>
        <select value={vial.resin} style={selS}
          onChange={e => { const m = e.target.value; set(m === "custom" ? { resin: m } : { resin: m, nResin: presets.resins[m] }); }}>
          {Object.keys(presets.resins).map(k => <option key={k} value={k}>{PRESET_LABEL[k] ?? k} · {presets.resins[k]}</option>)}
          <option value="custom">{PRESET_LABEL.custom}</option>
        </select>
      </div>
      {vial.resin === "custom" && <div style={fieldRow}><span style={fieldLbl}>n resina</span><NumField value={vial.nResin} min={1} max={3} step={0.001} onChange={v => set({ nResin: v })} /></div>}
      <div style={fieldRow} title="Absorción de la resina a la longitud de onda del proyector (cm⁻¹, Beer–Lambert). La luz se atenúa como exp(−α·L) desde la pared interior. 0 = sin absorción">
        <span style={fieldLbl}>Absorción α (cm⁻¹)</span>
        <NumField value={absorption} min={0} max={100} step={0.1} onChange={onAbsorption} />
      </div>
      {vial.enabled && (
        <>
          <div style={fieldRow} title="En el modelo: el CAL optimiza la imagen real del proyector con la refracción incluida (más margen de exposición). Posterior: se optimiza sin vial y luego se deforman las imágenes">
            <span style={fieldLbl}>Corrección</span>
            <Chips options={["model", "post"] as const} value={vial.mode ?? "model"} labels={{ model:"En el modelo", post:"Posterior" }}
              onChange={v => set({ mode: v })} />
          </div>
          {previewErr ? (
            <div style={{ marginTop:4, padding:"5px 8px", borderRadius:6, background:"#fef2f2", border:"1px solid #fecaca", fontSize:10.5, color:"#dc2626" }}>{previewErr}</div>
          ) : preview && (
            <div title="Índices aprox. a 405 nm; confírmalos con el fabricante. Supone proyector telecéntrico y vial centrado"
              style={{ marginTop:4, padding:"5px 8px", borderRadius:6, background: fits === false ? "#fef2f2" : "#eff6ff", border:`1px solid ${fits === false ? "#fecaca" : "#bfdbfe"}`, fontSize:10.5, color: fits === false ? "#dc2626" : "#1e40af", fontFamily:"'JetBrains Mono',monospace", lineHeight:1.5 }}>
              Zona imprimible Ø {preview.printable_diameter_mm.toFixed(2)} mm
              {partRadius != null && <> · pieza Ø {(2 * partRadius).toFixed(2)} {fits ? "✓" : "✗"}</>}
            </div>
          )}
        </>
      )}
    </Card>
  );
}

// ─── Transform Toolbar ────────────────────────────────────────────────────────

function TransformToolbar({ mode, setMode, onReset, outside, getTransform, applyOrientation, snap, setSnap, onAuto, autoBusy, autoMsg }: any) {
  const [rot, setRot] = useState({ rx:0, ry:0, rz:0 });
  
  useEffect(() => {
    const update = () => setRot(getTransform());
    update();
    window.addEventListener("vam-dirty", update);
    return () => window.removeEventListener("vam-dirty", update);
  }, [getTransform]);

  const hChange = (axis: "rx"|"ry"|"rz", v: string) => {
    const val = parseFloat(v); if(isNaN(val)) return;
    const n = { ...rot, [axis]: val };
    applyOrientation(n.rx, n.ry, n.rz);
    setRot(n);
  };

  return (
    <div style={{ position:"absolute", top:12, left:"50%", transform:"translateX(-50%)", display:"flex", gap:4, alignItems:"center", background:"rgba(255,255,255,0.94)", backdropFilter:"blur(8px)", border:"1px solid #e2e8f0", borderRadius:10, padding:"5px 8px", boxShadow:"0 2px 12px rgba(0,0,0,0.08)", zIndex:10 }}>
      {([
        { key:"rotate"    as const, icon:"↻", label:"Rotar"   },
      ]).map(t => (
        <button key={t.key} title={t.label} onClick={() => setMode(mode===t.key ? null : t.key)}
          style={{ width:32, height:32, border:"none", borderRadius:6, background:mode===t.key?"#eff6ff":"transparent", color:mode===t.key?"#2563eb":"#94a3b8", fontSize:16, cursor:"pointer", transition:"all 0.12s", display:"flex", alignItems:"center", justifyContent:"center" }}>
          {t.icon}
        </button>
      ))}
      <button title={snap?"Desactivar ajuste 5°":"Ajuste 5° (Snap)"} onClick={() => setSnap(!snap)} style={{ width:32, height:32, border:"none", borderRadius:6, background:snap?"#eff6ff":"transparent", color:snap?"#2563eb":"#94a3b8", fontSize:12, fontWeight:700, cursor:"pointer" }}>5°</button>
      <div style={{ width:1, height:20, background:"#e2e8f0", margin:"0 4px" }} />
      {["rx", "ry", "rz"].map(axis => (
        <div key={axis} style={{ display:"flex", alignItems:"center", gap:2 }}>
          <span style={{ fontSize:10, color:"#94a3b8", textTransform:"uppercase" }}>{axis[1]}</span>
          <input type="number" step={5} value={rot[axis as keyof typeof rot].toFixed(0)} onChange={e => hChange(axis as any, e.target.value)} style={{ width:38, height:24, fontSize:11, fontFamily:"'JetBrains Mono',monospace", border:"1px solid #e2e8f0", borderRadius:4, textAlign:"right", outline:"none", color:"#334155" }} />
        </div>
      ))}
      <div style={{ width:1, height:20, background:"#e2e8f0", margin:"0 4px" }} />
      <button title="Auto-orientación: la pieza lo más estrecha posible alrededor del eje de giro, sin pasar la altura ni el radio disponibles (solo geometría; no evalúa la dosis)"
        onClick={onAuto} disabled={autoBusy}
        style={{ height:32, padding:"0 10px", border:"none", borderRadius:6, background:"transparent", color:autoBusy?"#cbd5e1":"#2563eb", fontSize:11, fontWeight:700, cursor:autoBusy?"wait":"pointer" }}>
        {autoBusy ? "…" : "Auto"}
      </button>
      <button title="Reset" onClick={onReset} style={{ width:32, height:32, border:"none", borderRadius:6, background:"transparent", color:"#94a3b8", fontSize:14, cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center" }}>⟳</button>
      {autoMsg && <div style={{ marginLeft:4, fontSize:10, color:autoMsg.ok ? "#059669" : "#d97706", maxWidth:220, lineHeight:1.3 }}>{autoMsg.text}</div>}
      {outside && <div style={{ marginLeft:4, padding:"3px 10px", borderRadius:6, background:"#fef2f2", border:"1px solid #fecaca", fontSize:10, fontWeight:700, color:"#dc2626", letterSpacing:"0.05em" }}>FUERA DEL ÁREA</div>}
    </div>
  );
}

// ─── Library Card ─────────────────────────────────────────────────────────────

function LibRow({ item, onClick, onDelete }: { item: LibraryItem; onClick: () => void; onDelete: (e: React.MouseEvent) => void }) {
  return (
    <div onClick={onClick} style={{ 
      display:"grid", gridTemplateColumns:"64px 2fr 1fr 120px 120px 48px", alignItems:"center", gap:20, 
      padding:"12px 20px", background:"#fff", borderRadius:12, border:"1px solid #e2e8f0", 
      cursor:"pointer", transition:"all 0.15s", boxShadow:"0 1px 2px rgba(0,0,0,0.03)"
    }}
      onMouseEnter={e=>{const d=e.currentTarget as HTMLDivElement;d.style.boxShadow="0 4px 12px rgba(0,0,0,0.06)";d.style.borderColor="#cbd5e1";}}
      onMouseLeave={e=>{const d=e.currentTarget as HTMLDivElement;d.style.boxShadow="0 1px 2px rgba(0,0,0,0.03)";d.style.borderColor="#e2e8f0";}}
    >
      {/* Miniatura o Icono */}
      <div style={{ width:64, height:64, borderRadius:8, background:"#f1f5f9", overflow:"hidden", display:"flex", alignItems:"center", justifyContent:"center", border:"1px solid #f1f5f9" }}>
        {item.frame_sample 
          ? <img src={`${API}${item.frame_sample}`} style={{ width:"100%", height:"100%", objectFit:"cover", imageRendering:"pixelated" }} />
          : <span style={{ fontSize:20, opacity:0.2 }}>⬡</span>
        }
      </div>

      {/* Info Principal */}
      <div style={{ minWidth:0 }}>
        <div style={{ fontSize:14, fontWeight:700, color:"#1e293b", marginBottom:2, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{item.model_name}</div>
        <div style={{ fontSize:11, color:"#94a3b8", fontFamily:"'JetBrains Mono',monospace" }}>{fmtDt(item.timestamp)}</div>
      </div>

      {/* Parámetros */}
      <div style={{ fontSize:11, color:"#64748b" }}>
        <div style={{ opacity:0.6, fontSize:9, textTransform:"uppercase", letterSpacing:"0.05em" }}>{item.params.method || "CAL"} · Res/Ang</div>
        <div style={{ fontWeight:600 }}>{item.params.resolution}³ · {item.params.num_angles}°</div>
      </div>

      {/* Veredicto */}
      <div>
        <div style={{ display:"inline-flex", background:vb(item.verdict), color:vc(item.verdict), border:`1px solid ${vc(item.verdict)}33`, padding:"4px 10px", borderRadius:20, fontSize:10, fontWeight:700, letterSpacing:"0.05em" }}>
          {item.verdict}
        </div>
      </div>

      {/* Score */}
      <div>
        <div title={SCORE_TIP} style={{ fontSize:9, color:"#94a3b8", textTransform:"uppercase", letterSpacing:"0.05em", marginBottom:1, cursor:"help" }}>Score Bio · sim.</div>
        <div style={{ fontSize:22, fontWeight:800, color:vc(item.verdict), lineHeight:1 }}>{fmtScore(item.score_bio)}<span style={{ fontSize:11, opacity:0.7 }}>/100</span></div>
      </div>

      {/* Acciones */}
      <button onClick={onDelete} style={{ width:32, height:32, border:"none", borderRadius:8, background:"transparent", color:"#cbd5e1", fontSize:16, cursor:"pointer", transition:"all 0.15s", display:"flex", alignItems:"center", justifyContent:"center" }}
        onMouseEnter={e=>{const b=e.currentTarget as HTMLButtonElement;b.style.background="#fee2e2";b.style.color="#ef4444";}}
        onMouseLeave={e=>{const b=e.currentTarget as HTMLButtonElement;b.style.background="transparent";b.style.color="#cbd5e1";}}
      >🗑</button>
    </div>
  );
}

// ─── Detail View ──────────────────────────────────────────────────────────────

function DetailView({ item, onBack }: { item: LibraryItem; onBack: () => void }) {
  const [tab,       setTab]       = useState<"images"|"frames"|"metrics">("images");
  const [frameIdx,  setFrameIdx]  = useState(0);
  const [playing,   setPlaying]   = useState(false);
  const [activeImg, setActiveImg] = useState<string|null>(null);
  // Vista previa adaptativa: alto inicial = resolución del trabajo (512 → 512 px), limitado a lo que cabe en pantalla
  const maxZoom = Math.max(256, (typeof window !== "undefined" ? window.innerHeight : 900) - 230);
  const [zoomPx,    setZoomPx]    = useState(() => Math.min(maxZoom, Math.max(256, Number(item.params?.resolution) || 256)));
  const [native,    setNative]    = useState<{ w: number; h: number } | null>(null);
  const [frameMode, setFrameMode] = useState<"8bit" | "1bit">("1bit");
  const playRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  
  // Parámetros de impresión
  const [power, setPower] = useState(2.5);
  const [loops, setLoops] = useState(1);

  const activeFrames = frameMode === "1bit" && item.has_bitplanes
    ? (item.all_bitplanes ?? [])
    : (item.all_frames ?? []);

  useEffect(() => {
    if (playing && activeFrames.length > 0) playRef.current = setInterval(() => setFrameIdx(i => (i+1) % activeFrames.length), 33);
    else clearInterval(playRef.current);
    return () => clearInterval(playRef.current);
  }, [playing, activeFrames.length]);

  const staticImgs = [
    { label:"Reconstrucción XY / XZ / YZ", key:"reconstruction_slices" as const },
    { label:"Perfil por capa Z",            key:"layer_profile"         as const },
    { label:"Histograma de dosis",          key:"dose_histogram"        as const },
  ].filter(i => item.images[i.key]);

  return (
    <div style={{ position:"fixed", inset:0, background:"#f8fafc", zIndex:100, display:"grid", gridTemplateColumns:"1fr 340px", gridTemplateRows:"48px 1fr", overflow:"hidden" }}>
      {/* Topbar */}
      <div style={{ gridColumn:"1/-1", display:"flex", alignItems:"center", gap:10, padding:"0 20px", background:"#fff", borderBottom:"1px solid #e2e8f0", boxShadow:"0 1px 3px rgba(0,0,0,0.04)" }}>
        <button onClick={onBack} style={{ padding:"5px 12px", border:"1px solid #e2e8f0", borderRadius:6, background:"#f8fafc", color:"#475569", fontSize:12, cursor:"pointer", fontFamily:"inherit" }}>← Biblioteca</button>
        <div style={{ width:1, height:20, background:"#e2e8f0" }} />
        <span style={{ fontSize:13, fontWeight:600, color:"#0f172a", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{item.model_name}</span>
        <span style={{ fontSize:11, color:"#94a3b8" }}>{fmtDt(item.timestamp)}</span>
        <div title={SCORE_TIP} style={{ marginLeft:"auto", display:"inline-flex", background:vb(item.verdict), border:`1px solid ${vc(item.verdict)}44`, borderRadius:99, padding:"4px 14px", fontSize:11, fontWeight:700, color:vc(item.verdict), letterSpacing:"0.08em", textTransform:"uppercase", cursor:"help" }}>
          {item.verdict} · {fmtScore(item.score_bio)}/100 · sim.
        </div>
        <button onClick={() => window.open(`${API}/export/${item.job_id}.json`)} style={{ marginLeft:8, padding:"6px 16px", border:"1px solid #e2e8f0", borderRadius:6, background:"#fff", color:"#475569", fontSize:12, fontWeight:600, cursor:"pointer", fontFamily:"inherit" }}>📄 Exportar JSON</button>
      </div>

      {/* Main */}
      <div style={{ padding:"14px 20px", overflowY:tab==="metrics" ? "auto" : "hidden", display:"flex", flexDirection:"column", gap:12, minHeight:0 }}>
        {/* Tabs */}
        <div style={{ display:"flex", gap:4, background:"#f1f5f9", borderRadius:8, padding:4, width:"fit-content" }}>
          {(["images","frames","metrics"] as const).map(t => (
            <button key={t} onClick={() => setTab(t)} style={{ padding:"6px 16px", border:"none", borderRadius:6, background:tab===t?"#fff":"transparent", color:tab===t?"#0f172a":"#94a3b8", fontSize:12, fontWeight:tab===t?600:400, cursor:"pointer", fontFamily:"inherit", boxShadow:tab===t?"0 1px 3px rgba(0,0,0,0.08)":"none", transition:"all 0.12s" }}>
              {t==="images"?"Imágenes diagnóstico":t==="frames"?"Preview proyección":"Datos técnicos"}
            </button>
          ))}
        </div>

        {/* Imágenes */}
        {tab==="images" && (
          <div style={{ flex:1, minHeight:0, display:"grid", gridTemplateColumns:"1fr 1fr", gridTemplateRows:"minmax(0,1.2fr) minmax(0,1fr)", gap:10 }}>
            {staticImgs.map((img, i) => (
              <div key={img.key} onClick={() => setActiveImg(`${API}${item.images[img.key]}`)}
                style={{ gridColumn: i === 0 ? "1 / -1" : undefined, minHeight:0, display:"flex", flexDirection:"column", background:"#fff", border:"1px solid #e2e8f0", borderRadius:10, padding:"8px 10px", cursor:"zoom-in" }}>
                <div style={{ fontSize:10, color:"#94a3b8", textTransform:"uppercase", letterSpacing:"0.07em", marginBottom:4 }}>{img.label}</div>
                <div style={{ flex:1, minHeight:0, display:"flex", alignItems:"center", justifyContent:"center" }}>
                  <img src={`${API}${item.images[img.key]}`} style={{ maxWidth:"100%", maxHeight:"100%", objectFit:"contain", display:"block" }} />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Frames */}
        {tab==="frames" && (
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:10 }}>
            {item.has_bitplanes && (
              <div style={{ display:"flex", gap:6, marginBottom:10, justifyContent:"center" }}>
                {(["1bit","8bit"] as const).map(m => (
                  <button key={m} onClick={() => { setFrameMode(m); setFrameIdx(0); setPlaying(false); }}
                    style={{ padding:"4px 14px", border:"none", borderRadius:6, fontFamily:"inherit",
                             fontSize:11, fontWeight:frameMode===m?700:400, cursor:"pointer",
                             background:frameMode===m?"#2563eb":"#e2e8f0",
                             color:frameMode===m?"#fff":"#64748b" }}>
                    {m === "1bit"
                      ? `▶ Bitplanes 1-bit (${item.n_bitplanes} frames)`
                      : `8-bit original (${item.n_frames} frames)`}
                  </button>
                ))}
              </div>
            )}
            {frameMode === "1bit" && item.has_bitplanes && item.bitplane_params?.frame_w != null && (
              <div style={{ fontSize:10, color:"#64748b", textAlign:"center", marginBottom:8 }}>
                {item.bitplane_params.frame_w}×{item.bitplane_params.frame_h}px ·
                {item.bitplane_params.projector_pixel_um}µm/px ·
                {item.bitplane_params.rotation_speed_dps}°/s →
                {(360 / item.bitplane_params.rotation_speed_dps).toFixed(0)}s/vuelta ·
                SPI {item.bitplane_params.spi_ms_per_frame}ms/frame
              </div>
            )}
            {activeFrames.length === 0
              ? <div style={{ color:"#94a3b8", fontSize:13, padding:"60px 0", textAlign:"center" }}>No hay frames disponibles</div>
              : <>
                  <div style={{ background:"#111", borderRadius:10, overflow:"hidden", border:"1px solid #e2e8f0", height:zoomPx, width: native ? Math.round(zoomPx * native.w / native.h) : zoomPx, position:"relative", flexShrink:0 }}>
                    <img src={`${API}${activeFrames[frameIdx]}`}
                      onLoad={e => { const im = e.currentTarget; if (!native || native.w !== im.naturalWidth || native.h !== im.naturalHeight) setNative({ w: im.naturalWidth, h: im.naturalHeight }); }}
                      style={{ width:"100%", height:"100%", objectFit:"contain", display:"block", imageRendering:"pixelated" }} />
                    <div style={{ position:"absolute", bottom:8, right:8, background:"rgba(0,0,0,0.5)", color:"#fff", fontSize:10, padding:"2px 7px", borderRadius:4, fontFamily:"'JetBrains Mono',monospace" }}>
                      {frameIdx+1}/{activeFrames.length}
                    </div>
                  </div>
                  <div style={{ fontSize:10, color:"#94a3b8", textAlign:"center" }}>
                    Visualización {zoomPx}px de alto · resolución {item.params?.resolution ?? "—"}³ · fotograma nativo {native ? `${native.w}×${native.h}px` : "…"} para el proyector
                  </div>
                  {/* Zoom */}
                  <div style={{ display:"flex", alignItems:"center", gap:8, width:"100%", maxWidth:420 }}>
                    <span style={{ fontSize:10, color:"#94a3b8", minWidth:28 }}>128</span>
                    <input type="range" min={128} max={maxZoom} step={16} value={zoomPx} onChange={e => setZoomPx(Number(e.target.value))} style={{ flex:1 }} />
                    <span style={{ fontSize:10, color:"#94a3b8", minWidth:28 }}>{maxZoom}</span>
                    <span style={{ fontSize:11, color:"#334155", fontFamily:"'JetBrains Mono',monospace", minWidth:40 }}>{zoomPx}px</span>
                  </div>
                  {/* Controles */}
                  <div style={{ display:"flex", alignItems:"center", gap:8, width:"100%", maxWidth:420 }}>
                    <button onClick={() => setPlaying(p => !p)} style={{ padding:"7px 18px", border:"none", borderRadius:7, background:playing?"#f1f5f9":"#2563eb", color:playing?"#334155":"#fff", fontSize:12, fontWeight:600, cursor:"pointer", fontFamily:"inherit" }}>
                      {playing?"⏸ Pausar":"▶ Reproducir"}
                    </button>
                    <input type="range" min={0} max={activeFrames.length-1} value={frameIdx} onChange={e => { setPlaying(false); setFrameIdx(Number(e.target.value)); }} style={{ flex:1 }} />
                    <span style={{ fontSize:10, color:"#94a3b8", fontFamily:"'JetBrains Mono',monospace", minWidth:34 }}>
                      {((frameIdx/Math.max(activeFrames.length-1,1))*360).toFixed(0)}°
                    </span>
                  </div>
                </>
            }
          </div>
        )}

        {/* Métricas */}
        {tab==="metrics" && (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
            {[
              { title:"Calidad de dosis",  rows:[["VER",fmt4(item.metrics?.ver)],["CV",fmt4(item.metrics?.cv)],["PW",fmt4(item.metrics?.pw)],["IPDR",fmt4(item.metrics?.ipdr)],["Contraste",fmt2(item.metrics?.dose_contrast)]] },
              { title:"Dosis gel / void",  rows:[["Mín gel",fmt4(item.metrics?.min_gel_dose)],["Máx gel",fmt4(item.metrics?.max_gel_dose)],["Media gel",fmt4(item.metrics?.mean_gel_dose)],["Máx void",fmt4(item.metrics?.max_void_dose)],["Media void",fmt4(item.metrics?.mean_void_dose)]] },
              { title:"Análisis Z",        rows:[["Capas activas",String(item.metrics?.n_active_layers??"—")],["CV z p95",fmt4(item.metrics?.cv_z_p95)],["CV z worst",fmt4(item.metrics?.cv_z_worst)],["PW z p5",fmt4(item.metrics?.pw_z_p5)],["PW z worst",fmt4(item.metrics?.pw_z_worst)],["Solapamiento",`${fmt2(item.metrics?.pct_layers_overlap)}%`]] },
              { title:"Convergencia",      rows:[["Score Global",fmt2(item.score)],["Iteraciones",String(item.metrics?.n_valid_iters??"—")],["Conv. iter",item.metrics?.converged_iter===-1?"exit_param":String(item.metrics?.converged_iter??"—")],["Error inicial",fmt4(item.metrics?.initial_error)],["Error final",fmt4(item.metrics?.final_error)],["Mejora",fmt4(item.metrics?.error_improvement)]] },
              { title:"Orientación",       rows:[["Rx",`${item.orient?.rx?.toFixed(1)??"0.0"}°`],["Ry",`${item.orient?.ry?.toFixed(1)??"0.0"}°`],["Rz",`${item.orient?.rz?.toFixed(1)??"0.0"}°`]] },
              { title:"Configuración",     rows:[["Algoritmo",item.params?.method??"CAL"],["Resolución",`${item.params?.resolution??128}³`],["Ángulos",String(item.params?.num_angles??360)],["Filtro",String(item.params?.filter??"cosine")],["d_h",fmt4(item.params?.d_h as number)],["d_l",fmt4(item.params?.d_l as number)],["Tiempo CAL",fmts(item.metrics?.elapsed_time)],["Tiempo total",fmts(item.metrics?.total_time)]] },
            ].map(g => (
              <div key={g.title} style={{ background:"#fff", borderRadius:10, border:"1px solid #e2e8f0", padding:"13px 15px" }}>
                <div style={{ fontSize:10, color:"#94a3b8", textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:8, fontWeight:600 }}>{g.title}</div>
                {g.rows.map(([k,v]) => <Row key={k} label={k} value={v} />)}
              </div>
            ))}
            {item.penalties?.length > 0 && (
              <div style={{ gridColumn:"1/-1", background:"#fffbeb", borderRadius:10, border:"1px solid #fde68a", padding:"13px 15px" }}>
                <div style={{ fontSize:10, color:"#92400e", textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:8, fontWeight:600 }}>Penalizaciones</div>
                {item.penalties.map((p,i) => <div key={i} style={{ fontSize:12, color:"#92400e", padding:"4px 8px", background:"rgba(251,191,36,0.1)", borderRadius:4, borderLeft:"2px solid #fbbf24", marginBottom:4 }}>{p}</div>)}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right sidebar */}
      <div style={{ borderLeft:"1px solid #e2e8f0", background:"#f8fafc", display:"flex", flexDirection:"column", minHeight:0 }}>
        <div style={{ flex:1, overflowY:"auto", padding:"10px", display:"flex", flexDirection:"column", gap:8 }}>
          <ScoreBox verdict={item.verdict} score={item.score_bio} bd={item.score_breakdown}
                    tolPct={item.metrics?.exposure_tolerance_pct} rob={item.robustness} penalties={item.penalties} />
          <RobustBox tolPct={item.metrics?.exposure_tolerance_pct} rob={item.robustness} />
          <ProcessCard m={item.metrics} orient={item.orient} totalTime={item.total_time} frames={item.n_frames}
                       projector={item.params?.projector === "astra" ? "ASTRA (versión anterior)" : PROJECTOR_NAME} />
          <Card title="Ajustes de impresión">
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
              <div>
                <div style={{ fontSize:10, color:"#64748b", marginBottom:3 }} title="Pendiente calibrar a PWM">Potencia UV (mW/cm²)</div>
                <input type="number" step="0.1" value={power} onChange={e => setPower(Number(e.target.value))} style={{ width:"100%", padding:"6px 8px", border:"1px solid #e2e8f0", borderRadius:6, fontSize:12.5, fontFamily:"'JetBrains Mono',monospace", outline:"none" }} />
              </div>
              <div>
                <div style={{ fontSize:10, color:"#64748b", marginBottom:3 }}>Vueltas</div>
                <input type="number" step="1" min="1" value={loops} onChange={e => setLoops(Number(e.target.value))} style={{ width:"100%", padding:"6px 8px", border:"1px solid #e2e8f0", borderRadius:6, fontSize:12.5, fontFamily:"'JetBrains Mono',monospace", outline:"none" }} />
              </div>
            </div>
          </Card>
        </div>
        <div style={{ padding:"10px 12px 12px", borderTop:"1px solid #e2e8f0", background:"#fff" }}>
          <button onClick={() => window.dispatchEvent(new CustomEvent("vam-print", { detail: { id: item.job_id, item, power, loops } }))} style={{ width:"100%", padding:"11px", border:"none", borderRadius:8, background:"#0f172a", color:"#fff", fontFamily:"inherit", fontSize:13, fontWeight:600, letterSpacing:"0.06em", textTransform:"uppercase", cursor:"pointer" }}>▶ Enviar al proyector</button>
        </div>
      </div>

      {activeImg && (
        <div onClick={() => setActiveImg(null)} style={{ position:"fixed", inset:0, background:"rgba(0,0,0,0.85)", zIndex:200, display:"flex", alignItems:"center", justifyContent:"center", cursor:"zoom-out" }}>
          <img src={activeImg} style={{ maxWidth:"90vw", maxHeight:"90vh", borderRadius:8 }} />
        </div>
      )}
    </div>
  );
}

// ─── Library View ─────────────────────────────────────────────────────────────

function LibraryView({ onSelect }: { onSelect: (i: LibraryItem) => void }) {
  const [items,   setItems]   = useState<LibraryItem[]>([]);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    fetch(`${API}/library`).then(r => r.json()).then(d => { setItems(d.items ?? []); setLoading(false); }).catch(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm("¿Eliminar permanentemente este registro?")) return;
    try {
      const res = await fetch(`${API}/library/${id}`, { method: "DELETE" });
      if (res.ok) load();
    } catch (err) {
      console.error("Delete failed", err);
    }
  };

  return (
    <div style={{ padding:"28px 32px", overflowY:"auto", height:"100%", background:"#f8fafc" }}>
      <div style={{ display:"flex", alignItems:"baseline", gap:12, marginBottom:24, maxWidth:1000, margin:"0 auto 24px" }}>
        <h2 style={{ fontSize:18, fontWeight:700, color:"#0f172a" }}>Biblioteca</h2>
        <span style={{ fontSize:12, color:"#94a3b8" }}>{items.length} modelos guardados</span>
      </div>
      {loading && <div style={{ display:"flex", alignItems:"center", justifyContent:"center", height:"60%", color:"#94a3b8", fontSize:13 }}>Cargando…</div>}
      {!loading && items.length === 0 && (
        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", height:"60%", gap:12 }}>
          <div style={{ fontSize:40, opacity:0.1 }}>📦</div>
          <div style={{ fontSize:13, color:"#94a3b8", textAlign:"center" }}>Aún no hay modelos guardados.<br/>Usa el botón "Guardar" tras una optimización.</div>
        </div>
      )}
      {!loading && items.length > 0 && (
        <div style={{ display:"flex", flexDirection:"column", gap:12, maxWidth:1000, margin:"0 auto" }}>
          {items.map(i => <LibRow key={i.job_id} item={i} onClick={() => onSelect(i)} onDelete={e => handleDelete(e, i.job_id)} />)}
        </div>
      )}
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

type AppView = "optimizer" | "library" | "detail";

export default function App() {
  // Canvas ref — el canvas vive en un div position:fixed SIEMPRE presente
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { loadSTL, setTransformMode, resetTransform, setOutsideCb, setSnap, getPartRadius,
          getPrintRotation, applyPrintRotation, applyPrintRotationMatrix } = useThreeScene(canvasRef);
  const [autoBusy, setAutoBusy] = useState(false);
  const [autoMsg, setAutoMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const [isSnap, setSnapStateRaw] = useState(false);
  const setSnapState = useCallback((s: boolean) => { setSnapStateRaw(s); setSnap(s); }, [setSnap]);

  const [view,      setView]      = useState<AppView>("optimizer");
  const [detail,    setDetail]    = useState<LibraryItem|null>(null);
  const [fileName,  setFileName]  = useState("");
  const [fileData,  setFileData]  = useState<File|null>(null);
  const [jobId,     setJobId]     = useState("");
  const [jobStatus, setJobStatus] = useState<JobStatus|null>(null);
  const [polling,   setPolling]   = useState(false);
  const [dragging,  setDragging]  = useState(false);
  const [useCuda,   setUseCuda]   = useState(true);
  const [startTime, setStartTime] = useState(0);
  const [tMode,     setTModeRaw]  = useState<"rotate"|null>(null);
  const [outside,   setOutside]   = useState(false);
  const [cfg,       setCfg]       = useState<Config>(DEF_CFG);
  const [vial,      setVial]      = useState<VialCfg>(DEF_VIAL);
  const [vialPresets, setVialPresets] = useState(FALLBACK_PRESETS);
  const [vialPreview, setVialPreview] = useState<VialPreview | null>(null);
  const [vialErr,   setVialErr]   = useState<string | null>(null);
  const [partRadius, setPartRadius] = useState<number | null>(null);
  const [isSaved,   setIsSaved]   = useState(false);
  const [isDirty,   setIsDirty]   = useState(false);
  const [currStep,  setCurrStep]  = useState("");
  const [isPrinting, setIsPrinting] = useState(false);
  const [rpiStatus,  setRpiStatus]  = useState<RPiStatus | null>(null);
  const [printPhase, setPrintPhase] = useState("Preparando...");
  const [printItem,  setPrintItem]  = useState<LibraryItem | null>(null);
  const [printStart, setPrintStart] = useState(0);
  const [printElapsed, setPrintElapsed] = useState(0);
  const [pollErrors,   setPollErrors]   = useState(0);
  const [showConnection, setShowConnection] = useState(false);
  const [cm4, setCm4] = useState<CM4DiscoverResult | null>(null);
  const [discoveringCm4, setDiscoveringCm4] = useState(false);
  const [cm4Error, setCm4Error] = useState<string | null>(null);
  const [printConfirm, setPrintConfirm] = useState<{id: string, item: LibraryItem | null, power: number, loops: number} | null>(null);
  // BUG-02: timestamp de inicio de impresión para ignorar estado FINALIZADO stale
  const printStartedAtRef = useRef<number>(0);

  // BUG-08: Al arrancar, comprobar si hay una impresión activa en la CM4 (sesión recuperada)
  useEffect(() => {
    const checkActiveprint = async () => {
      try {
        const r = await fetch(`${API}/printer_status`, { signal: AbortSignal.timeout(2000) });
        if (!r.ok) return;
        const d = await r.json() as RPiStatus;
        if (d.status === "PROYECTANDO" || d.status === "CARGANDO_MODELO") {
          // Hay una impresión en curso de una sesión anterior — retomar seguimiento
          setRpiStatus(d);
          setPrintPhase(d.status === "PROYECTANDO" ? "projecting" : "cm4_loading");
          setPrintStart(Date.now());
          setPrintElapsed(0);
          setIsPrinting(true);
          printStartedAtRef.current = Date.now();
        }
      } catch { /* CM4 no disponible al arrancar, ignorar */ }
    };
    checkActiveprint();
  }, []);

  const handleDiscoverCm4 = async () => {
    setDiscoveringCm4(true);
    setCm4Error(null);
    try {
      const r = await fetch(`${API}/cm4/discover`);
      const d: CM4DiscoverResult = await r.json();
      if (d.found) {
        setCm4(d);
      } else {
        setCm4(null);
        setCm4Error("Placa no encontrada. Verifica el cable USB.");
      }
    } catch {
      setCm4(null);
      setCm4Error("No se pudo conectar con el servidor local.");
    } finally { 
      setDiscoveringCm4(false); 
    }
  };

  const handleDisconnectCm4 = () => {
    setCm4(null);
    setCm4Error(null);
  };

  useEffect(() => { setOutsideCb(setOutside); }, []);

  // Presets de materiales del vial y de resinas (si el servidor no responde, se usan los de respaldo)
  useEffect(() => {
    fetch(`${API}/vial/presets`).then(r => r.ok ? r.json() : null).then(d => { if (d) setVialPresets(d); }).catch(() => {});
  }, []);

  // Zona imprimible en vivo (cálculo analítico en el servidor, sin GPU), con debounce
  useEffect(() => {
    if (!vial.enabled) { setVialPreview(null); setVialErr(null); return; }
    const q = new URLSearchParams({ outer_diameter_mm: String(vial.outerD), wall_thickness_mm: String(vial.wall) });
    if (vial.material === "custom") q.set("n_vial", String(vial.nVial)); else q.set("material", vial.material);
    if (vial.resin === "custom") q.set("n_resin", String(vial.nResin)); else q.set("resin", vial.resin);
    const id = setTimeout(() => {
      fetch(`${API}/vial/preview?${q}`)
        .then(async r => { const d = await r.json(); if (!r.ok) throw new Error(d.detail ?? "Parámetros no válidos"); return d; })
        .then(d => { setVialPreview(d); setVialErr(null); })
        .catch(e => { setVialPreview(null); setVialErr(e.message === "Failed to fetch" ? "Sin conexión con el servidor" : e.message); });
    }, 250);
    return () => clearTimeout(id);
  }, [vial]);

  // Radio de la pieza respecto al eje de giro: se recalcula al cargar o girar el modelo
  useEffect(() => {
    const update = () => setPartRadius(getPartRadius());
    update();
    window.addEventListener("vam-dirty", update);
    return () => window.removeEventListener("vam-dirty", update);
  }, [getPartRadius, fileData]);

  const vialBlocks = vial.enabled && (!!vialErr || (vialPreview != null && partRadius != null && partRadius > vialPreview.printable_radius_mm));

  const setTMode = useCallback((m: "rotate"|null) => {
    setTModeRaw(m);
    setTransformMode(m);
  }, [setTransformMode]);

  useEffect(() => {
    const fn = () => setIsDirty(true);
    window.addEventListener("vam-dirty", fn);
    
    const onPrint = async (e: any) => {
      const id = e.detail.id ?? e.detail;
      const item: LibraryItem | null = e.detail.item ?? null;
      const power = e.detail.power ?? 2.5;
      const loops = e.detail.loops ?? 1;
      setPrintConfirm({ id, item, power, loops });
    };
    window.addEventListener("vam-print", onPrint);

    return () => {
      window.removeEventListener("vam-dirty", fn);
      window.removeEventListener("vam-print", onPrint);
    };
  }, []);

  const handleConfirmPrint = async () => {
    if (!printConfirm) return;
    const { id, item, power, loops } = printConfirm;
    setPrintConfirm(null);
    
    const startedAt = Date.now();
    printStartedAtRef.current = startedAt;
    setIsPrinting(true);
    setPrintPhase("zipping");
    setPrintItem(item);
    setPrintStart(startedAt);
    setPrintElapsed(0);
    setRpiStatus(null);
    setPollErrors(0);
    try {
      const res = await fetch(`${API}/print/${id}?power=${power}&loops=${loops}`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json();
        alert(`Error al enviar: ${data.error || 'Desconocido'}`);
        setIsPrinting(false);
      } else {
        setPrintPhase("cm4_loading");
      }
    } catch (error) {
      alert("Error de conexión con el backend local.");
      setIsPrinting(false);
    }
  };

  // Desactivar gizmo al salir del optimizador
  useEffect(() => {
    if (view !== "optimizer") {
      setTransformMode(null);
      setTModeRaw(null);
    }
  }, [view]);

  // Polling
  useEffect(() => {
    if (!polling || !jobId) return;
    const id = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/status/${jobId}`);
        const data = await res.json() as JobStatus;
        setJobStatus(data);
        if (data.status === "done") {
          setPolling(false);
        }
        if (data.status === "error") setPolling(false);
      } catch { /* servidor no disponible */ }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [polling, jobId]);

  // Polling RPi + elapsed timer
  useEffect(() => {
    if (!isPrinting) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API}/printer_status`, { signal: AbortSignal.timeout(2000) });
        if (!res.ok) throw new Error("CM4 Offline");
        const data = await res.json() as RPiStatus;

        setRpiStatus(data);
        setPollErrors(0);

        // BUG-02: ignorar FINALIZADO si llegó antes de que el nuevo trabajo arrancase
        // (estado residual de la impresión anterior mientras la CM4 descomprime el ZIP)
        const isStaleFinished = data.status === "FINALIZADO" && (printPhase === "zipping" || printPhase === "cm4_loading");
        if (isStaleFinished) return;

        // Transiciones de fase basadas en el estado del hardware
        if (data.status === "CARGANDO_MODELO") setPrintPhase("cm4_loading");
        if (data.status === "PROYECTANDO")     setPrintPhase("projecting");
        if (data.status === "FINALIZADO")      setPrintPhase("done");
        if (data.status === "ERROR")           setPrintPhase("error");
        
        // Si el estado es IDLE pero estábamos proyectando, es que terminó
        if (data.status === "IDLE" && printPhase === "projecting") {
          setPrintPhase("done");
        }

        if (data.status !== "FINALIZADO" && data.status !== "ERROR" && data.status !== "IDLE") {
          setPrintElapsed(Math.floor((Date.now() - printStart) / 1000));
        }
      } catch (e) {
        const newErrors = pollErrors + 1;
        setPollErrors(newErrors);
        setPrintElapsed(Math.floor((Date.now() - printStart) / 1000));
        // BUG-04: si la CM4 no responde durante 10s seguidos, declarar error
        if (newErrors >= 10 && printPhase !== "done" && printPhase !== "error") {
          setPrintPhase("error");
        }
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [isPrinting, printStart, printPhase, pollErrors]);

  const handleFile = useCallback((file: File) => {
    if (!file.name.toLowerCase().endsWith(".stl")) return;
    setFileName(file.name); setFileData(file); setJobStatus(null);
    const reader = new FileReader();
    reader.onload = e => { if (e.target?.result instanceof ArrayBuffer) loadSTL(e.target.result); };
    reader.readAsArrayBuffer(file);
  }, [loadSTL]);

  // Auto-orientación (POST /orient): giro en el sistema de impresión, con los límites del vial si está activo
  const handleAutoOrient = async () => {
    if (!fileData || autoBusy) return;
    setAutoBusy(true); setAutoMsg(null);
    const form = new FormData();
    form.append("file", fileData);
    if (vial.enabled) {
      form.append("vial_outer_diameter_mm", String(vial.outerD));
      form.append("vial_wall_thickness_mm", String(vial.wall));
      if (vial.material === "custom") form.append("n_vial", String(vial.nVial)); else form.append("vial_material", vial.material);
      if (vial.resin === "custom") form.append("n_resin", String(vial.nResin)); else form.append("resin", vial.resin);
    }
    try {
      const res = await fetch(`${API}/orient`, { method:"POST", body:form });
      const d = await res.json();
      if (!res.ok) throw new Error(d.detail ?? res.statusText);
      applyPrintRotationMatrix(d.rotation);
      const lim = d.radius_limit_source === "vial" ? "zona imprimible del vial" : "área de trabajo";
      setAutoMsg(d.feasible
        ? { ok:true, text: d.reduction_pct > 0.05
            ? `Rx ${d.rx.toFixed(1)}° · Ry ${d.ry.toFixed(1)}° · ancho −${d.reduction_pct.toFixed(0)} %`
            : "Ya estaba en su mejor orientación" }
        : { ok:false, text:`No cabe en la ${lim} (radio ${d.radius_mm.toFixed(1)} mm, alto ${d.height_mm.toFixed(1)} mm)` });
    } catch (e: any) {
      setAutoMsg({ ok:false, text:`Auto-orientación falló: ${e?.message ?? e}` });
    } finally {
      setAutoBusy(false);
    }
  };

  const handleRun = async () => {
    if (!fileData || outside || vialBlocks) return;
    // Giro en el sistema del STL (el que aplica cal_core), no los ángulos del visor
    const { rx, ry, rz } = getPrintRotation();
    const form = new FormData();
    form.append("file", fileData);
    form.append("cuda", String(useCuda));
    form.append("rx", String(rx));
    form.append("ry", String(ry));
    form.append("rz", String(rz));
    form.append("num_angles",  String(cfg.numAngles));
    form.append("resolution",  String(cfg.resolution));
    form.append("filter",      cfg.filter);
    form.append("method",      cfg.method);
    const margin = (cfg.stop ?? "ver") === "margin";
    form.append("stop_ver_mask", margin ? "guard_band" : "half");
    if (margin) form.append("stop_pw_margin", String(cfg.pwMargin ?? 0.03));
    if ((cfg.absorptionCm ?? 0) > 0) form.append("absorption_cm", String(cfg.absorptionCm));
    if (vial.enabled) {
      form.append("vial_outer_diameter_mm", String(vial.outerD));
      form.append("vial_wall_thickness_mm", String(vial.wall));
      if (vial.material === "custom") form.append("n_vial", String(vial.nVial)); else form.append("vial_material", vial.material);
      if (vial.resin === "custom") form.append("n_resin", String(vial.nResin)); else form.append("resin", vial.resin);
      form.append("vial_mode", vial.mode ?? "model");
    }
    try {
      const res  = await fetch(`${API}/optimize`, { method:"POST", body:form });
      const data = await res.json();
      setJobId(data.job_id);
      setStartTime(Date.now());
      setJobStatus({ job_id:data.job_id, status:"running", progress:"Iniciando…" });
      setPolling(true);
      setTMode(null);
      setIsSaved(false);
      setIsDirty(false);
      setCurrStep("Iniciando conexión…");
    } catch {
      setJobStatus({ job_id:"", status:"error", progress:"", error:"No se puede conectar con el servidor." });
    }
  };

  const handleSave = async () => {
    if (!jobId) return;
    try {
      const res = await fetch(`${API}/save/${jobId}`, { method: "POST" });
      if (res.ok) setIsSaved(true);
    } catch (e) {
      console.error("Save failed", e);
    }
  };

  const isRunning = jobStatus?.status === "running";
  const isDone    = jobStatus?.status === "done";
  const result    = jobStatus?.summary;

  // ── Render ──────────────────────────────────────────────────────────────────
  // El canvas está en un div position:fixed que NUNCA desaparece del DOM.
  // Las vistas de biblioteca y detalle son overlays con z-index mayor.
  return (
    <div style={{ fontFamily:"'DM Sans',system-ui,sans-serif", color:"#334155" }}>

      {/* ════ CANVAS — siempre montado, nunca se desmonta ════════════════ */}
      <div style={{
        position: "fixed",
        top:    TOP_H,
        left:   LEFT_W,
        right:  RIGHT_W,
        bottom: 0,
        zIndex: 1,
        background: "#f8fafc",
      }}>
        <canvas ref={canvasRef} style={{ width:"100%", height:"100%", display:"block" }} />

        {/* Transform toolbar — solo visible en optimizer con modelo cargado */}
        {view === "optimizer" && fileName && (
          <TransformToolbar
            mode={tMode}
            setMode={setTMode}
            onReset={() => { resetTransform(); setTModeRaw(null); setOutside(false); }}
            outside={outside}
            getTransform={getPrintRotation}
            applyOrientation={applyPrintRotation}
            onAuto={handleAutoOrient}
            autoBusy={autoBusy}
            autoMsg={autoMsg}
            snap={isSnap}
            setSnap={setSnapState}
          />
        )}

        <div style={{ position:"absolute", bottom:16, left:16, fontSize:10, color:"#94a3b8", lineHeight:1.8, fontFamily:"'JetBrains Mono',monospace", pointerEvents:"none" }}>
          <div>Ø20 × 25 mm</div>
          <div style={{ color:"#cbd5e1" }}>Drag · Scroll</div>
        </div>

        {/* Barra de progreso en el borde inferior */}
        {isRunning && (
          <div style={{ position:"absolute", bottom:0, left:0, right:0, height:2, background:"#e2e8f0" }}>
            <div style={{ height:"100%", background:"#2563eb", animation:"progress 2s ease-in-out infinite" }} />
          </div>
        )}
      </div>

      {/* ════ TOPBAR — siempre montado ═══════════════════════════════════ */}
      <div style={{
        position: "fixed", top:0, left:0, right:0, height:TOP_H, zIndex:30,
        display:"flex", alignItems:"center", gap:12, padding:"0 20px",
        background:"#fff", borderBottom:"1px solid #e2e8f0",
        boxShadow:"0 1px 3px rgba(0,0,0,0.04)",
      }}>
        <div style={{ display:"flex", alignItems:"center", gap:7 }}>
          <div style={{ width:7, height:7, borderRadius:"50%", background:"#2563eb" }} />
          <span style={{ fontSize:13, fontWeight:700, letterSpacing:"0.08em", textTransform:"uppercase", color:"#0f172a" }}>VAMCORE</span>
        </div>
        <div style={{ display:"flex", gap:2, marginLeft:14 }}>
          {(["optimizer","library"] as const).map(v => (
            <button key={v} onClick={() => setView(v)} style={{ padding:"4px 14px", border:"none", borderRadius:6, background:view===v?"#eff6ff":"transparent", color:view===v?"#2563eb":"#94a3b8", fontSize:12, fontWeight:view===v?600:400, cursor:"pointer", fontFamily:"inherit", transition:"all 0.12s" }}>
              {v === "optimizer" ? "Optimizador" : "Biblioteca"}
            </button>
          ))}
        </div>
        <span style={{ fontSize:11, color:"#94a3b8", marginLeft:4, fontFamily:"'JetBrains Mono',monospace" }}>
          {cfg.method} · {PROJECTOR_NAME} · {cfg.resolution}³ · {cfg.numAngles}°
        </span>
        {isRunning && <span style={{ marginLeft:"auto", fontSize:11, color:"#2563eb" }}>● Optimizando…</span>}
        {isPrinting && <span style={{ marginLeft:"auto", fontSize:11, color:"#ef4444", fontWeight:700 }}>● IMPRIMIENDO…</span>}
        {/* Conexión directa en Header (Simple) */}
        <div style={{ marginLeft: isRunning || isPrinting ? 8 : "auto", display: "flex", alignItems: "center", gap: 6 }}>
          {cm4?.found ? (
            <>
              <div style={{ padding: "5px 12px", background: "#dcfce7", color: "#166534", borderRadius: 6, fontSize: 11, fontWeight: 800, letterSpacing: "0.05em" }}>
                CONECTADO
              </div>
              <button 
                onClick={handleDisconnectCm4}
                style={{ padding: "5px 10px", border: "none", background: "transparent", color: "#94a3b8", fontSize: 10, fontWeight: 600, cursor: "pointer", textTransform: "uppercase" }}
                onMouseEnter={e => e.currentTarget.style.color = "#ef4444"}
                onMouseLeave={e => e.currentTarget.style.color = "#94a3b8"}
              >
                Desconectar
              </button>
            </>
          ) : (
            <div style={{ position: "relative", display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}>
              <button 
                onClick={handleDiscoverCm4} 
                disabled={discoveringCm4} 
                style={{ 
                  padding: "6px 14px", borderRadius: 6, 
                  border: cm4Error ? "1px solid #fecaca" : "none",
                  background: discoveringCm4 ? "#e2e8f0" : cm4Error ? "#fef2f2" : "#2563eb", 
                  color: discoveringCm4 ? "#94a3b8" : cm4Error ? "#dc2626" : "#fff", 
                  fontSize: 11, fontWeight: 800, letterSpacing: "0.05em", cursor: discoveringCm4 ? "wait" : "pointer",
                  transition: "all 0.2s",
                }}
              >
                {discoveringCm4 ? "CONECTANDO..." : cm4Error ? "REINTENTAR" : "CONECTAR"}
              </button>
              {cm4Error && (
                <span style={{ fontSize: 9, color: "#dc2626", fontWeight: 600, whiteSpace: "nowrap", maxWidth: 160, textAlign: "right" }}>
                  {cm4Error}
                </span>
              )}
            </div>

          )}
        </div>
      </div>



      {/* ════ LEFT PANEL — siempre montado ═══════════════════════════════ */}
      <div style={{
        position:"fixed", top:TOP_H, left:0, bottom:0, width:LEFT_W, zIndex:20,
        background:"#f8fafc", borderRight:"1px solid #e2e8f0", display:"flex", flexDirection:"column",
        // Ocultar visualmente cuando no estamos en optimizer (pero NO desmontar)
        pointerEvents: view === "optimizer" ? "auto" : "none",
        opacity: view === "optimizer" ? 1 : 0,
        transition: "opacity 0.15s",
      }}>
        <div style={{ flex:1, overflowY:"auto", padding:"12px 12px 8px", display:"flex", flexDirection:"column", gap:10 }}>
          <Card title="Modelo" right={<span style={{ fontSize:9.5, color:"#cbd5e1", fontFamily:"'JetBrains Mono',monospace" }}>área Ø20 × 25 mm</span>}>
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]); }}
              onClick={() => document.getElementById("stl-input")?.click()}
              style={{ border:`1.5px dashed ${dragging?"#2563eb":"#cbd5e1"}`, borderRadius:8, padding:"10px", textAlign:"center", cursor:"pointer", background:dragging?"#eff6ff":"#f8fafc", transition:"all 0.15s", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}
            >
              <input id="stl-input" type="file" accept=".stl" onChange={e => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }} style={{ display:"none" }} />
              {fileName
                ? <><span style={{ fontSize:13, color:"#059669" }}>✓</span><span style={{ fontSize:11.5, color:"#334155", wordBreak:"break-all" }}>{fileName}</span><span style={{ fontSize:10, color:"#94a3b8" }}>· cambiar</span></>
                : <><span style={{ fontSize:15, color:"#94a3b8" }}>↑</span><span style={{ fontSize:11.5, color:"#94a3b8" }}>Arrastra un .stl o haz clic</span></>
              }
            </div>
          </Card>

          <VialPanel vial={vial} onChange={setVial} presets={vialPresets} preview={vialPreview} previewErr={vialErr} partRadius={partRadius}
                     absorption={cfg.absorptionCm ?? 0}
                     onAbsorption={v => { setCfg({ ...cfg, absorptionCm: v }); window.dispatchEvent(new CustomEvent("vam-dirty")); }} />

          <ConfigPanel cfg={cfg} onChange={setCfg} useCuda={useCuda} setUseCuda={setUseCuda} />

          {outside && <div style={{ padding:"7px 10px", borderRadius:6, background:"#fef2f2", border:"1px solid #fecaca", fontSize:11, color:"#dc2626", fontWeight:600 }}>⚠ Modelo fuera del área de trabajo</div>}
          {vialBlocks && <div style={{ padding:"7px 10px", borderRadius:6, background:"#fef2f2", border:"1px solid #fecaca", fontSize:11, color:"#dc2626", fontWeight:600 }}>⚠ {vialErr ?? "La pieza no cabe en la zona imprimible del vial: reoriéntala o usa un vial mayor"}</div>}
          {jobStatus?.status==="error" && <div style={{ padding:"7px 10px", borderRadius:6, background:"#fef2f2", border:"1px solid #fecaca", fontSize:11, color:"#dc2626" }}>{jobStatus.error}</div>}
        </div>

        {/* Acción principal, siempre visible */}
        <div style={{ padding:"10px 12px 12px", borderTop:"1px solid #e2e8f0", background:"#fff", display:"flex", flexDirection:"column", gap:6 }}>
          {isRunning ? (
            <>
              <ProgressButton
                startTime={startTime}
                totalSec={Math.max(15, 80 * Math.pow(cfg.resolution / 128, 3) * (cfg.numAngles / 360))}
                onStep={setCurrStep}
              />
              <div style={{ fontSize:10, color:"#ef4444", textAlign:"center", fontWeight:600, letterSpacing:"0.04em", textTransform:"uppercase", whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis", padding:"0 4px" }}>{currStep}</div>
            </>
          ) : isDone && !isDirty ? (
            <button disabled style={{ width:"100%", padding:"11px", border:"none", borderRadius:8, background:"#059669", color:"#fff", fontFamily:"inherit", fontSize:13, fontWeight:800, letterSpacing:"0.06em", cursor:"default" }}>
              PROCESADO ✓
            </button>
          ) : (
            <button onClick={handleRun} disabled={!fileData||outside||vialBlocks} style={{ width:"100%", padding:"11px", border:"none", borderRadius:8, background:(!fileData||outside||vialBlocks)?"#f1f5f9":"#2563eb", color:(!fileData||outside||vialBlocks)?"#cbd5e1":"#fff", fontFamily:"inherit", fontSize:13, fontWeight:800, letterSpacing:"0.06em", textTransform:"uppercase", cursor:(!fileData||outside||vialBlocks)?"not-allowed":"pointer", transition:"all 0.15s", boxShadow:(!fileData||outside||vialBlocks)?"none":"0 4px 12px rgba(37,99,235,0.2)" }}>
              PROCESAR
            </button>
          )}
          <div style={{ fontSize:9.5, color:"#94a3b8", textAlign:"center", fontFamily:"'JetBrains Mono',monospace" }}>
            CAL · {PROJECTOR_NAME} · {cfg.resolution}³ · {cfg.numAngles}° · {useCuda ? "GPU" : "CPU"}
          </div>
        </div>
      </div>

      {/* ════ RIGHT PANEL — siempre montado ══════════════════════════════ */}
      <div style={{
        position:"fixed", top:TOP_H, right:0, bottom:0, width:RIGHT_W, zIndex:20,
        background:"#f8fafc", borderLeft:"1px solid #e2e8f0", display:"flex", flexDirection:"column",
        pointerEvents: view === "optimizer" ? "auto" : "none",
        opacity: view === "optimizer" ? 1 : 0,
        transition: "opacity 0.15s",
      }}>
        <div style={{ flex:1, overflowY:"auto", padding:"10px", display:"flex", flexDirection:"column", gap:8 }}>
          {!isDone && !isRunning && (
            <div style={{ flex:1, display:"flex", alignItems:"center", justifyContent:"center", color:"#cbd5e1", fontSize:12, textAlign:"center", lineHeight:2.2 }}>
              Carga un STL<br/>y pulsa Procesar<br/>para ver resultados
            </div>
          )}
          {isRunning && (
            <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:14 }}>
              <div style={{ width:32, height:32, borderRadius:"50%", border:"2px solid #e2e8f0", borderTopColor:"#2563eb", animation:"spin 0.8s linear infinite" }} />
              <span style={{ fontSize:11, color:"#94a3b8" }}>Procesando…</span>
            </div>
          )}
          {isDone && result && (
            <>
              <ScoreBox verdict={result.verdict} score={result.score_bio} bd={result.score_breakdown}
                        tolPct={result.metrics?.exposure_tolerance_pct} rob={result.robustness} penalties={result.penalties} />
              <RobustBox tolPct={result.metrics?.exposure_tolerance_pct} rob={result.robustness} />
              <DoseCard m={result.metrics} />
              <VialCard v={result.vial} />
              <ProcessCard m={result.metrics} orient={result.orient} totalTime={result.total_time} frames={cfg.numAngles}
                           projector={PROJECTOR_NAME} />
            </>
          )}
        </div>
        {isDone && result && (
          <div style={{ padding:"10px 12px 12px", borderTop:"1px solid #e2e8f0", background:"#fff" }}>
            {!isSaved ? (
              <button onClick={handleSave} style={{ width:"100%", padding:"10px", border:"none", borderRadius:8, background:"#059669", color:"#fff", fontFamily:"inherit", fontSize:12, fontWeight:700, cursor:"pointer", boxShadow:"0 2px 6px rgba(5,150,105,0.2)" }}>
                💾 GUARDAR EN BIBLIOTECA · {cfg.numAngles} fotogramas
              </button>
            ) : (
              <button onClick={() => setView("library")} style={{ width:"100%", padding:"10px", border:"1px solid #e2e8f0", borderRadius:8, background:"#eff6ff", color:"#2563eb", fontFamily:"inherit", fontSize:12, fontWeight:700, cursor:"pointer" }}>
                ✓ GUARDADO · VER BIBLIOTECA
              </button>
            )}
          </div>
        )}
      </div>

      {/* ════ OVERLAY — Biblioteca ════════════════════════════════════════ */}
      {view === "library" && (
        <div style={{ position:"fixed", inset:0, zIndex:50, background:"#f8fafc", display:"flex", flexDirection:"column" }}>
          {/* Topbar de biblioteca (replica el nav) */}
          <div style={{ height:TOP_H, display:"flex", alignItems:"center", gap:12, padding:"0 20px", background:"#fff", borderBottom:"1px solid #e2e8f0", boxShadow:"0 1px 3px rgba(0,0,0,0.04)", flexShrink:0 }}>
            <div style={{ display:"flex", alignItems:"center", gap:7 }}>
              <div style={{ width:7, height:7, borderRadius:"50%", background:"#2563eb" }} />
              <span style={{ fontSize:13, fontWeight:700, letterSpacing:"0.08em", textTransform:"uppercase", color:"#0f172a" }}>VAMCORE</span>
            </div>
            <div style={{ display:"flex", gap:2, marginLeft:14 }}>
              {(["optimizer","library"] as const).map(v => (
                <button key={v} onClick={() => setView(v)} style={{ padding:"4px 14px", border:"none", borderRadius:6, background:view===v?"#eff6ff":"transparent", color:view===v?"#2563eb":"#94a3b8", fontSize:12, fontWeight:view===v?600:400, cursor:"pointer", fontFamily:"inherit", transition:"all 0.12s" }}>
                  {v === "optimizer" ? "Optimizador" : "Biblioteca"}
                </button>
              ))}
            </div>
          </div>
          <div style={{ flex:1, overflow:"hidden" }}>
            <LibraryView onSelect={i => { setDetail(i); setView("detail"); }} />
          </div>
        </div>
      )}

      {/* ════ OVERLAY — Detalle ══════════════════════════════════════════ */}
      {view === "detail" && detail && <DetailView item={detail} onBack={() => setView("library")} />}

      {/* ════ OVERLAY — PRINTER MODAL ════════════════════════════════════ */}
      {isPrinting && (() => {
        // --- Cálculo de tiempo estimado de impresión ---
        const bp = printItem?.bitplane_params;
        const totalFrames = bp?.n_bitplanes ?? printItem?.n_bitplanes ?? rpiStatus?.total_frames ?? 0;
        const msPerFrame  = bp?.spi_ms_per_frame ?? 0;
        const estimatedPrintSec = msPerFrame > 0 && totalFrames > 0 ? (totalFrames * msPerFrame) / 1000 : 0;

        const isProjecting = printPhase === "projecting";
        const isDonePhase  = printPhase === "done";
        const isErrorPhase = printPhase === "error";
        const isPreparing  = !isProjecting && !isDonePhase && !isErrorPhase;
        
        let barPct = 0;
        if (isDonePhase || isErrorPhase) barPct = 100;
        else if (isProjecting && rpiStatus?.progress_percent) barPct = rpiStatus.progress_percent;
        else if (isPreparing) barPct = 0;
        
        const cm4Offline   = pollErrors >= 3;

        // Tiempo restante estimado (solo en fase proyectando)
        const projRemSec  = estimatedPrintSec > 0 && isProjecting
          ? Math.max(0, Math.round(estimatedPrintSec - (rpiStatus?.elapsed_s ?? 0)))
          : null;
        const fmtSec = (s: number) => s < 60 ? `${s}s` : `${Math.floor(s/60)}m ${s%60}s`;

        return (
          <div style={{ position:"fixed", inset:0, zIndex:100, background:"rgba(15,23,42,0.9)", backdropFilter:"blur(8px)", display:"flex", alignItems:"center", justifyContent:"center", padding:20 }}>
            <div style={{ width:500, background:"#fff", borderRadius:24, overflow:"hidden", boxShadow:"0 25px 50px -12px rgba(0,0,0,0.5)", display:"flex", flexDirection:"column", position: "relative" }}>
              
              {/* Botón de cierre forzado (X) */}
              <button 
                onClick={() => { setIsPrinting(false); setPrintPhase("zipping"); setPrintItem(null); setPollErrors(0); }}
                style={{ position: "absolute", top: 16, right: 16, border: "none", background: "rgba(0,0,0,0.05)", width: 32, height: 32, borderRadius: "50%", cursor: "pointer", fontSize: 16, color: "#94a3b8", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 10, transition: "all 0.2s" }}
                onMouseEnter={e => e.currentTarget.style.background = "rgba(220,38,38,0.1)"}
                onMouseLeave={e => e.currentTarget.style.background = "rgba(0,0,0,0.05)"}
              >
                ✕
              </button>

              {/* Header de Estado */}
              <div style={{ padding:"45px 40px 30px", textAlign:"center", background: isDonePhase ? "#f0fdf4" : isErrorPhase ? "#fef2f2" : "#ffffff" }}>
                {isPreparing && (
                  <div style={{ fontSize:15, fontWeight:700, color:"#2563eb", letterSpacing:"0.15em", textTransform:"uppercase", marginBottom:16 }}>
                    <span style={{ display:"inline-block", animation:"ping 1.5s infinite", marginRight:8 }}>●</span>
                    Preparando Hardware
                  </div>
                )}
                {isProjecting && (
                  <div style={{ fontSize:15, fontWeight:700, color:"#2563eb", letterSpacing:"0.15em", textTransform:"uppercase", marginBottom:16 }}>
                    <span style={{ display:"inline-block", animation:"ping 1.5s infinite", marginRight:8 }}>●</span>
                    Proyectando Capas
                  </div>
                )}
                {isDonePhase && (
                  <div style={{ fontSize:15, fontWeight:700, color:"#16a34a", letterSpacing:"0.15em", textTransform:"uppercase", marginBottom:16 }}>
                    ✓ Impresión Finalizada
                  </div>
                )}
                {isErrorPhase && (
                  <div style={{ fontSize:15, fontWeight:700, color:"#dc2626", letterSpacing:"0.15em", textTransform:"uppercase", marginBottom:16 }}>
                    ❌ Error de Impresión
                  </div>
                )}

                {/* Main Stats (Percentage & Frame) */}
                <div style={{ display:"flex", alignItems:"baseline", justifyContent:"center", gap:16, marginBottom:20 }}>
                  {isPreparing ? (
                     <div style={{ fontSize:32, fontWeight:800, color:"#0f172a", fontFamily:"'JetBrains Mono',monospace", lineHeight:1, letterSpacing:"0.05em" }}>
                       {printPhase === "zipping" ? "COMPRIMIENDO..." : "ENVIANDO DATOS..."}
                     </div>
                  ) : (
                     <div style={{ fontSize:80, fontWeight:800, color:isDonePhase?"#166534":isErrorPhase?"#991b1b":"#0f172a", fontFamily:"'JetBrains Mono',monospace", lineHeight:1 }}>
                       {Math.round(barPct)}<span style={{ fontSize:40, color:"#94a3b8" }}>%</span>
                     </div>
                  )}
                </div>

                {/* Subinfo */}
                <div style={{ fontSize:16, color:"#64748b", fontFamily:"'JetBrains Mono',monospace" }}>
                   {isProjecting && rpiStatus && (
                     `Frame ${rpiStatus.current_frame} de ${rpiStatus.total_frames}`
                   )}
                   {isDonePhase && (
                     `Completada exitosamente`
                   )}
                   {isErrorPhase && (
                     `Fallo en el backend`
                   )}
                   {isPreparing && (
                     `Espera por favor`
                   )}
                </div>
              </div>

              {/* Progress Bar Container */}
              <div style={{ padding:"0 40px" }}>
                <div style={{ height:16, background:"#f1f5f9", borderRadius:8, overflow:"hidden", position:"relative" }}>
                   {isPreparing ? (
                     <div style={{ position:"absolute", inset:0, background:"linear-gradient(90deg,transparent,#2563eb,transparent)", animation:"indeterminate 1.4s ease-in-out infinite", backgroundSize:"200% 100%" }} />
                   ) : (
                     <div style={{ height:"100%", background: isDonePhase ? "#059669" : isErrorPhase ? "#ef4444" : "#2563eb", width:`${barPct}%`, transition:"width 0.6s ease-out", borderRadius:8 }} />
                   )}
                </div>
              </div>

              {/* Time Stats */}
              <div style={{ padding:"35px 40px", display:"flex", justifyContent:"space-between" }}>
                <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                  <span style={{ fontSize:12, color:"#94a3b8", textTransform:"uppercase", letterSpacing:"0.1em", fontWeight:700 }}>Tiempo Transcurrido</span>
                  <span style={{ fontSize:32, fontWeight:800, color:"#334155", fontFamily:"'JetBrains Mono',monospace" }}>{fmtSec(printElapsed)}</span>
                </div>
                {!isDonePhase && !isErrorPhase && (
                  <div style={{ display:"flex", flexDirection:"column", gap:6, alignItems:"flex-end" }}>
                    <span style={{ fontSize:12, color:"#94a3b8", textTransform:"uppercase", letterSpacing:"0.1em", fontWeight:700 }}>Tiempo Restante</span>
                    <span style={{ fontSize:32, fontWeight:800, color:"#2563eb", fontFamily:"'JetBrains Mono',monospace" }}>
                      {projRemSec !== null ? `~${fmtSec(projRemSec)}` : "—"}
                    </span>
                  </div>
                )}
              </div>


              {/* Footer / Botón */}
              <div style={{ padding:"24px 40px 32px", borderTop:"1px solid #f1f5f9", display:"flex", justifyContent:"center" }}>
                {isDonePhase || isErrorPhase ? (
                  <button onClick={() => { setIsPrinting(false); setPrintPhase("zipping"); setPrintItem(null); setPollErrors(0); }}
                    style={{ width:"100%", padding:"16px", background:isDonePhase?"#059669":"#ef4444", color:"#fff", border:"none", borderRadius:12, fontSize:15, fontWeight:800, cursor:"pointer", textTransform:"uppercase", letterSpacing:"0.05em", boxShadow:isDonePhase?"0 4px 14px rgba(5,150,105,0.3)":"none" }}>
                    Cerrar Ventana
                  </button>
                ) : (
                  <div style={{ fontSize:13, color:"#94a3b8", textAlign:"center", fontWeight:500 }}>
                    🔒 No cierres la aplicación hasta finalizar la impresión
                  </div>
                )}
              </div>

            </div>
          </div>
        );
      })()}

      {/* Modal Confirmación Impresión (Reemplaza a window.confirm) */}
      {printConfirm && !isPrinting && (
        <div style={{ position:"fixed", inset:0, background:"rgba(15,23,42,0.8)", zIndex:1000, display:"flex", alignItems:"center", justifyContent:"center" }}>
          <div style={{ background:"#fff", width:440, borderRadius:12, padding:24, boxShadow:"0 20px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)" }}>
            <h3 style={{ margin:"0 0 16px 0", fontSize:18, color:"#0f172a" }}>Confirmar Impresión</h3>
            <p style={{ margin:"0 0 24px 0", fontSize:14, color:"#475569", lineHeight:1.5 }}>
              Vas a enviar el modelo <strong>{printConfirm.item?.model_name || printConfirm.id}</strong> a la impresora vía SPI.<br/><br/>
              Ajustes de impresión: <strong>{printConfirm.power} mW/cm²</strong> · <strong>{printConfirm.loops} Vueltas</strong><br/><br/>
              ¿Deseas encender la UV e iniciar la proyección volumétrica ahora mismo?
            </p>
            <div style={{ display:"flex", gap:12, justifyContent:"flex-end" }}>
              <button onClick={() => setPrintConfirm(null)} style={{ padding:"10px 18px", border:"1px solid #e2e8f0", borderRadius:6, background:"#f8fafc", color:"#475569", fontSize:13, fontWeight:600, cursor:"pointer" }}>
                Cancelar
              </button>
              <button onClick={handleConfirmPrint} style={{ padding:"10px 18px", border:"none", borderRadius:6, background:"#0f172a", color:"#fff", fontSize:13, fontWeight:600, cursor:"pointer", boxShadow:"0 4px 6px -1px rgba(0,0,0,0.1)" }}>
                ▶ Iniciar Proyección
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:#e2e8f0; border-radius:2px; }
        @keyframes spin         { to { transform:rotate(360deg); } }
        @keyframes spinSlow     { to { transform:rotate(360deg); } }
        @keyframes progress     { 0%{width:0%;margin-left:0} 50%{width:60%;margin-left:20%} 100%{width:0%;margin-left:100%} }
        @keyframes shimmer      { 0%{transform:translateX(-100%)} 100%{transform:translateX(100%)} }
        @keyframes indeterminate{ 0%{background-position:200% 0} 100%{background-position:-200% 0} }
        @keyframes ping         { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.4)} }
      `}</style>
    </div>
  );
}
