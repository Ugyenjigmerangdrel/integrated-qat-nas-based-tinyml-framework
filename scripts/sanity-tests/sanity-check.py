# sanity_check.py
import sys, os, tempfile, json, traceback
from importlib import import_module

RESULTS = {}

def ok(name, detail="ok"):
    RESULTS[name] = {"status": "PASS", "detail": detail}

def fail(name, err):
    RESULTS[name] = {"status": "FAIL", "detail": f"{err.__class__.__name__}: {err}"}
    traceback.print_exc()

def section(title):
    print("\n" + "="*8 + f" {title} " + "="*8)

# 1) TensorFlow + TFMOT
section("TensorFlow & Model Optimization")
try:
    import tensorflow as tf
    v = getattr(tf, "__version__", "unknown")
    # Eager mode & simple op
    tf.constant([1.0]) + tf.constant([2.0])
    # Tiny Keras model forward
    m = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    _ = m(tf.constant([[3.0]]))
    # GPU presence
    gpus = tf.config.list_physical_devices("GPU")
    gpu_msg = f"GPUs: {len(gpus)} -> {[d.name for d in gpus]}" if gpus else "No GPU visible"
    ok("tensorflow", f"{v}; {gpu_msg}")
except Exception as e:
    fail("tensorflow", e)

try:
    import tensorflow_model_optimization as tfmot
    v = getattr(tfmot, "__version__", "unknown")
    ok("tensorflow-model-optimization", v)
except Exception as e:
    fail("tensorflow-model-optimization", e)

# 2) Flatbuffers
section("Flatbuffers")
try:
    import flatbuffers
    v = getattr(flatbuffers, "__version__", "unknown")
    # Quick sanity: build a tiny buffer (no schema) and ensure bytes object exists
    b = flatbuffers.Builder(0)
    s = b.CreateString("hello")
    b.Finish(s)
    _ = bytes(b.Output())
    ok("flatbuffers", v)
except Exception as e:
    fail("flatbuffers", e)

# 3) NumPy / SciPy / pandas / scikit-learn
section("NumPy / SciPy / pandas / scikit-learn")
try:
    import numpy as np
    v = np.__version__
    a = np.array([[1.0, 2.0],[3.0, 4.0]])
    _ = a @ a
    ok("numpy", v)
except Exception as e:
    fail("numpy", e)

try:
    import scipy
    from scipy import linalg
    v = scipy.__version__
    _ = linalg.inv([[1.0, 0.0],[0.0, 2.0]])
    ok("scipy", v)
except Exception as e:
    fail("scipy", e)

try:
    import pandas as pd
    v = pd.__version__
    df = pd.DataFrame({"x":[1,2,3], "y":[3,2,1]})
    _ = df.describe()
    ok("pandas", v)
except Exception as e:
    fail("pandas", e)

try:
    import sklearn
    from sklearn.linear_model import LogisticRegression
    v = sklearn.__version__
    X = [[0.0],[1.0],[2.0],[3.0]]
    y = [0,0,1,1]
    clf = LogisticRegression(max_iter=100).fit(X,y)
    _ = clf.predict([[1.5]])
    ok("scikit-learn", v)
except Exception as e:
    fail("scikit-learn", e)

# 4) Matplotlib (headless save)
section("Matplotlib")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = matplotlib.__version__
    plt.figure()
    plt.plot([0,1,2],[0,1,0])
    tmp = os.path.join(tempfile.gettempdir(), "mpl_test.png")
    plt.savefig(tmp)
    ok("matplotlib", f"{v}; saved {tmp}")
except Exception as e:
    fail("matplotlib", e)

# 5) JupyterLab import/version
section("JupyterLab")
try:
    jlab = import_module("jupyterlab")
    v = getattr(jlab, "__version__", "unknown")
    ok("jupyterlab", v)
except Exception as e:
    fail("jupyterlab", e)

# 6) tqdm / rich
section("tqdm / rich")
try:
    from tqdm import tqdm
    for _ in tqdm(range(3), desc="tqdm-check", leave=False):
        pass
    ok("tqdm", "ok")
except Exception as e:
    fail("tqdm", e)

try:
    from rich.console import Console
    from rich.table import Table
    con = Console()
    tbl = Table(title="rich-check"); tbl.add_column("k"); tbl.add_column("v")
    tbl.add_row("ping","pong")
    con.print(tbl)
    ok("rich", "ok")
except Exception as e:
    fail("rich", e)

# 7) ONNX / onnxsim / onnxruntime
section("ONNX / onnxsim / onnxruntime")
try:
    import onnx
    from onnx import helper, TensorProto, checker
    v = onnx.__version__
    # tiny model: y = x + x
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1,2])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1,2])
    node = helper.make_node("Add", inputs=["x","x"], outputs=["y"])
    graph = helper.make_graph([node], "add_graph", [X], [Y])
    model = helper.make_model(graph, producer_name="sanity")
    checker.check_model(model)
    onnx_path = os.path.join(tempfile.gettempdir(), "add.onnx")
    onnx.save(model, onnx_path)
    ok("onnx", f"{v}; saved {onnx_path}")
except Exception as e:
    fail("onnx", e)

try:
    import onnxsim
    from onnxsim import simplify
    # Simplify the model we just saved
    m = onnx.load(onnx_path)
    sm, check = simplify(m)
    ok("onnxsim", f"check={check}")
except Exception as e:
    fail("onnxsim", e)

try:
    import onnxruntime as ort
    v = ort.__version__
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    import numpy as np
    out = sess.run(None, {"x": np.array([[1.0, 2.0]], dtype=np.float32)})
    assert out[0].shape == (1,2)
    ok("onnxruntime", v)
except Exception as e:
    fail("onnxruntime", e)

# 8) Pillow
section("Pillow")
try:
    from PIL import Image, ImageDraw
    import PIL
    v = PIL.__version__
    img = Image.new("RGB", (64,64), (10,120,200))
    d = ImageDraw.Draw(img); d.text((5,25), "OK", fill=(255,255,255))
    p = os.path.join(tempfile.gettempdir(), "pil_test.jpg")
    img.save(p)
    ok("pillow", f"{v}; saved {p}")
except Exception as e:
    fail("pillow", e)

# Summary
section("SUMMARY")
# Try rich table if available; else plain text
try:
    from rich.console import Console
    from rich.table import Table
    con = Console()
    tbl = Table(title="Library Sanity Summary")
    tbl.add_column("Package")
    tbl.add_column("Status")
    tbl.add_column("Detail", overflow="fold")
    for k in sorted(RESULTS.keys()):
        tbl.add_row(k, RESULTS[k]["status"], RESULTS[k]["detail"])
    con.print(tbl)
except Exception:
    for k in sorted(RESULTS.keys()):
        print(f"{k:28} {RESULTS[k]['status']:5}  {RESULTS[k]['detail']}")

# Exit code: 0 if all pass, else 1
exit_code = 0 if all(v["status"]=="PASS" for v in RESULTS.values()) else 1
print("\nExit code:", exit_code)
sys.exit(exit_code)
