"""
ML Model — EfficientNetV2 with Transfer Learning
Industry-grade inference with caching, Grad-CAM, and error handling.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

DISEASE_METADATA: Dict[str, Dict] = {
    "Tomato___Late_blight": {
        "display_name": "Tomato Late Blight",
        "severity": "HIGH",
        "immediate_action": "Immediately remove and destroy infected plant parts.",
        "fungicide": "Chlorothalonil 2.5g/L or Mancozeb 2g/L",
        "prevention": "Improve air circulation; avoid overhead watering.",
        "affected_crop": "Tomato",
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato Early Blight",
        "severity": "MEDIUM",
        "immediate_action": "Remove lower infected leaves.",
        "fungicide": "Copper-based fungicide every 7-10 days.",
        "prevention": "Rotate crops; stake plants for airflow.",
        "affected_crop": "Tomato",
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato Leaf Mold",
        "severity": "MEDIUM",
        "immediate_action": "Reduce humidity; remove affected leaves.",
        "fungicide": "Chlorothalonil or Copper fungicide.",
        "prevention": "Use resistant varieties; maintain 60% humidity.",
        "affected_crop": "Tomato",
    },
    "Potato___Late_blight": {
        "display_name": "Potato Late Blight",
        "severity": "HIGH",
        "immediate_action": "Apply fungicide immediately.",
        "fungicide": "Metalaxyl + Mancozeb 2.5g/L",
        "prevention": "Plant certified disease-free tubers.",
        "affected_crop": "Potato",
    },
    "Pepper,_bell___Bacterial_spot": {
        "display_name": "Pepper Bacterial Spot",
        "severity": "MEDIUM",
        "immediate_action": "Remove infected fruits and leaves.",
        "fungicide": "Copper bactericide spray.",
        "prevention": "Use disease-free seed; avoid overhead watering.",
        "affected_crop": "Pepper",
    },
    "Corn_(maize)___Common_rust_": {
        "display_name": "Corn Common Rust",
        "severity": "MEDIUM",
        "immediate_action": "Apply fungicide at first sign of pustules.",
        "fungicide": "Azoxystrobin or Propiconazole.",
        "prevention": "Plant rust-resistant hybrids.",
        "affected_crop": "Corn",
    },
    "Apple___Apple_scab": {
        "display_name": "Apple Scab",
        "severity": "MEDIUM",
        "immediate_action": "Prune and destroy infected leaves.",
        "fungicide": "Captan or Myclobutanil at green tip.",
        "prevention": "Rake fallen leaves; plant resistant varieties.",
        "affected_crop": "Apple",
    },
    "healthy": {
        "display_name": "Healthy Plant",
        "severity": "NONE",
        "immediate_action": "No action needed.",
        "fungicide": "None required.",
        "prevention": "Continue good agricultural practices.",
        "affected_crop": "N/A",
    },
}


class PlantDiseaseModel:
    def __init__(self):
        self._model = None
        self._labels: List[str] = []
        self._is_loaded = False
        self._is_demo = False

    def load(self, model_path: str, labels_path: str) -> None:
        try:
            import tensorflow as tf

            if Path(labels_path).exists():
                with open(labels_path) as f:
                    self._labels = json.load(f)
            else:
                self._labels = list(DISEASE_METADATA.keys())

            if Path(model_path).exists():
                self._model = tf.keras.models.load_model(model_path)
                self._is_demo = False
                logger.info("Loaded trained model from disk", path=model_path)
            else:
                # No trained weights — pure demo mode, no inference
                self._model = None
                self._is_demo = True
                logger.warning("No trained model found — DEMO MODE active", path=model_path)

            self._is_loaded = True
            logger.info("Model ready", classes=len(self._labels), demo=self._is_demo)

        except ImportError:
            logger.warning("TensorFlow not installed — demo mode")
            self._labels = list(DISEASE_METADATA.keys())
            self._is_demo = True
            self._is_loaded = True

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        import tensorflow as tf
        img = tf.image.resize(image, (224, 224))
        img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
        return np.expand_dims(img, axis=0)

    def predict(self, image: np.ndarray) -> Dict:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._is_demo or self._model is None:
            return self._demo_prediction()

        processed = self.preprocess(image)
        probs = self._model.predict(processed, verbose=0)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        pred_class = self._labels[top3_idx[0]]
        confidence = float(probs[top3_idx[0]])

        if confidence < 0.1:
            logger.warning("Confidence too low — model likely untrained, using demo")
            return self._demo_prediction()

        meta = DISEASE_METADATA.get(pred_class, DISEASE_METADATA["healthy"])
        return {
            "class_key": pred_class,
            "display_name": meta["display_name"],
            "confidence": round(confidence, 4),
            "is_healthy": "healthy" in pred_class.lower(),
            "severity": meta["severity"],
            "treatment": {
                "immediate_action": meta["immediate_action"],
                "fungicide": meta["fungicide"],
                "prevention": meta["prevention"],
            },
            "affected_crop": meta["affected_crop"],
            "top3": [
                {
                    "class_key": self._labels[idx],
                    "display_name": DISEASE_METADATA.get(self._labels[idx], {}).get("display_name", self._labels[idx]),
                    "confidence": round(float(probs[idx]), 4),
                }
                for idx in top3_idx
            ],
            "demo_mode": False,
        }

    def _demo_prediction(self) -> Dict:
        import random
        key = random.choice(list(DISEASE_METADATA.keys()))
        meta = DISEASE_METADATA[key]
        other_keys = [k for k in DISEASE_METADATA.keys() if k != key]
        conf = round(random.uniform(0.78, 0.97), 4)
        top3 = [{"class_key": key, "display_name": meta["display_name"], "confidence": conf}]
        for k in random.sample(other_keys, min(2, len(other_keys))):
            top3.append({
                "class_key": k,
                "display_name": DISEASE_METADATA[k]["display_name"],
                "confidence": round(random.uniform(0.01, 0.12), 4),
            })
        return {
            "class_key": key,
            "display_name": meta["display_name"],
            "confidence": conf,
            "is_healthy": "healthy" in key.lower(),
            "severity": meta["severity"],
            "treatment": {
                "immediate_action": meta["immediate_action"],
                "fungicide": meta["fungicide"],
                "prevention": meta["prevention"],
            },
            "affected_crop": meta["affected_crop"],
            "top3": top3,
            "demo_mode": True,
        }

    def generate_gradcam(self, image: np.ndarray, layer_name: str = "top_conv") -> Optional[np.ndarray]:
        if self._model is None:
            return None
        try:
            import tensorflow as tf
            grad_model = tf.keras.Model(
                inputs=self._model.inputs,
                outputs=[self._model.get_layer(layer_name).output, self._model.output],
            )
            processed = self.preprocess(image)
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(processed)
                pred_idx = tf.argmax(predictions[0])
                loss = predictions[:, pred_idx]
            grads = tape.gradient(loss, conv_outputs)[0]
            weights = tf.reduce_mean(grads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
            cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
            cam = cam.numpy()
            cam_resized = np.array(tf.image.resize(cam[..., np.newaxis], (224, 224))).squeeze()
            return cam_resized
        except Exception as e:
            logger.warning("Grad-CAM failed", error=str(e))
            return None


@lru_cache(maxsize=1)
def get_model() -> PlantDiseaseModel:
    from core.config import settings
    model = PlantDiseaseModel()
    model.load(settings.MODEL_PATH, settings.LABELS_PATH)
    return model