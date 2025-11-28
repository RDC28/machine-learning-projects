# app.py
from flask import Flask, render_template, request, abort, redirect, url_for
from dotenv import load_dotenv
import os, json, pickle, numpy as np
from datetime import datetime

load_dotenv()
PORT = os.getenv('PORT', '5000')

app = Flask(__name__)

# load model metadata once at startup
MODEL_INFO_PATH = os.path.join(os.path.dirname(__file__), 'model_info.json')
with open(MODEL_INFO_PATH, 'r', encoding='utf-8') as f:
    MODEL_INFO = json.load(f)

def get_model_meta(key):
    return MODEL_INFO.get(key)

def get_default_model_key():
    # choose a default model to land users on; prefer explicit 'default' key if present
    for k, v in MODEL_INFO.items():
        if v.get('default', False):
            return k
    # fallback to first key if any
    try:
        return next(iter(MODEL_INFO))
    except StopIteration:
        return None

# inject common template context (e.g., current year)
@app.context_processor
def inject_common():
    return {"current_year": datetime.utcnow().year}

@app.route('/')
def home():
    # Group models by type
    grouped = {}
    for key, meta in MODEL_INFO.items():
        model_type = meta.get("type", "other").capitalize()
        if model_type not in grouped:
            grouped[model_type] = []
        grouped[model_type].append({
            "key": key,
            "display_name": meta.get("display_name"),
            "short_description": meta.get("short_description"),
            "subtype": meta.get("subtype", "").capitalize()
        })

    return render_template('index.html', grouped_models=grouped)

@app.route('/prediction')
def prediction_root():
    """Redirect the old /prediction URL to a default model page so the navbar 'Predict' link always works."""
    default_key = get_default_model_key()
    if not default_key:
        # no models available — show a simple page (or 404)
        abort(404, description="No models available")
    return redirect(url_for('model_predict', model_key=default_key))

@app.route('/models/<model_key>', methods=['GET', 'POST'])
def model_predict(model_key):
    meta = get_model_meta(model_key)
    if not meta:
        abort(404, description=f"Model '{model_key}' not found.")

    if request.method == 'POST':
        # read inputs in the order defined by meta['parameters']
        try:
            raw_inputs = []
            for p in meta.get('parameters', []):
                val = request.form.get(p['name'])
                if val is None:
                    raise ValueError(f"Missing parameter: {p['name']}")
                raw_inputs.append(float(val))

            # prepare paths
            model_path = os.path.join(os.path.dirname(__file__), meta['model_path'])
            scaler_path = os.path.join(os.path.dirname(__file__), meta.get('scaler_path', ''))

            # ensure model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # load model
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)

            # prepare input array
            inp = np.array([raw_inputs])
            X = inp
            loaded_scaler = None
            if scaler_path and os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    loaded_scaler = pickle.load(f)
                try:
                    X = loaded_scaler.transform(inp)
                except Exception:
                    # fallback to raw inputs if scaler transform fails
                    X = inp

            # predict
            res = loaded_model.predict(X)

            # robust extraction of scalar result
            try:
                arr = np.asarray(res)
                result_raw = arr.flatten()[0]
            except Exception:
                result_raw = res

            # try converting to numeric when possible
            numeric_result = None
            try:
                numeric_result = float(result_raw)
            except Exception:
                numeric_result = None

            # apply server-side rounding if requested and numeric
            result_value = result_raw
            if numeric_result is not None and meta.get("round") is not None:
                try:
                    result_value = round(numeric_result, int(meta["round"]))
                except Exception:
                    result_value = numeric_result
            elif numeric_result is not None:
                result_value = numeric_result

            # build JSON payload
            result_json = {"prediction": result_value}
            display_label = None

            # labels mapping (dynamic): uses meta["labels"] if present
            labels_map = meta.get("labels")
            if labels_map:
                # try to map numeric prediction to label (common case for classifiers)
                label = None
                try:
                    # if prediction is numeric (possibly float), round to nearest int key
                    if numeric_result is not None:
                        pred_key = int(round(numeric_result))
                    else:
                        pred_key = str(result_raw)
                except Exception:
                    pred_key = str(result_raw)

                # lookup by string key (JSON keys are strings)
                label = labels_map.get(str(pred_key))
                if label is None and isinstance(pred_key, int):
                    # try non-string key if someone used int keys in memory
                    label = labels_map.get(pred_key)

                if label is not None:
                    display_label = label
                    result_json["label"] = label

            # optional probabilities
            if meta.get("probability") and hasattr(loaded_model, "predict_proba"):
                try:
                    Xp = X if 'Xp' not in locals() else Xp
                    probs = loaded_model.predict_proba(Xp)[0].tolist()
                    result_json["probabilities"] = probs
                except Exception:
                    # ignore probability failures
                    pass

            # pass sanitized values to template
            return render_template(
                'result.html',
                result=result_value,
                model=meta,
                inputs=raw_inputs,
                display_label=display_label,
                result_json=result_json
            )

        except Exception as e:
            # handle parsing / model errors gracefully (could log)
            return render_template('result.html', error=str(e), model=meta, inputs=None)

    # GET -> show form based on parameters
    return render_template('prediction.html', model=meta)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(PORT))
