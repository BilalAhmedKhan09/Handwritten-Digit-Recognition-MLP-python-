from IPython.display import display, HTML
from google.colab import output as colab_output
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io, base64, ipywidgets as widgets
from IPython.display import clear_output

out = widgets.Output()

def preprocess_array(arr):
    if arr.mean() > 128:
        arr = 255.0 - arr
    arr = np.where(arr > 30, arr, 0.0)
    rows, cols = np.any(arr > 0, axis=1), np.any(arr > 0, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        arr = arr[r0:r1+1, c0:c1+1]
    pad = max(arr.shape) // 4
    arr = np.pad(arr, pad, constant_values=0)
    arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((28,28)), dtype=float) / 255.0
    return arr

def predict_and_show(arr28):
    pred  = model.predict([arr28.flatten()])[0]
    proba = model.predict_proba([arr28.flatten()])[0]
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].imshow(arr28.reshape(28,28), cmap='gray')
    axes[0].set_title(f'What the model sees\nPredicted: {pred}', fontsize=14)
    axes[0].axis('off')
    colors = ['steelblue']*10; colors[pred] = 'green'
    axes[1].bar(range(10), proba, color=colors)
    axes[1].set_xticks(range(10))
    axes[1].set_xlabel('Digit'); axes[1].set_ylabel('Confidence')
    axes[1].set_title('Prediction Confidence per Digit')
    plt.tight_layout(); plt.show()
    print(f'✅ Predicted: {pred}  (confidence: {proba[pred]*100:.1f}%)')

def predict_canvas(data_url):
    with out:
        clear_output(wait=True)
        _, encoded = data_url.split(',', 1)
        img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('L')
        predict_and_show(preprocess_array(np.array(img, dtype=float)))

colab_output.register_callback('predict_canvas', predict_canvas)

# Upload handler
def on_predict_upload(b):
    with out:
        clear_output(wait=True)
        if not upload_btn.value:
            print('⚠️ Upload an image first.'); return
        img = Image.open(io.BytesIO(list(upload_btn.value.values())[0]['content'])).convert('L')
        predict_and_show(preprocess_array(np.array(img, dtype=float)))

upload_btn     = widgets.FileUpload(accept='image/*', multiple=False,
                                    layout=widgets.Layout(width='200px'))
upload_pred_btn = widgets.Button(description='🔍 Predict Image',
                                  button_style='success',
                                  layout=widgets.Layout(width='180px'))
upload_pred_btn.on_click(on_predict_upload)


display(HTML("""
<div style="font-family:sans-serif">
  <h3>✏️ Draw a digit (0–9) below:</h3>
  <canvas id="dc" width="300" height="300"
    style="border:3px solid #444; border-radius:8px;
           background:#000; cursor:crosshair; display:block;"></canvas>
  <div style="margin-top:10px">
    <button onclick="clearDC()"
      style="padding:8px 20px; font-size:14px; border-radius:6px;
             margin-right:10px; cursor:pointer;">🗑️ Clear</button>
    <button onclick="sendDC()"
      style="padding:8px 20px; font-size:14px; border-radius:6px;
             background:#4CAF50; color:white; border:none; cursor:pointer;">
      🔍 Predict</button>
  </div>
  <p id="msg" style="color:gray; margin-top:8px">Draw a digit, then click Predict.</p>
</div>

<script>
  const cv  = document.getElementById('dc');
  const ctx = cv.getContext('2d');
  ctx.strokeStyle = 'white';
  ctx.lineWidth   = 22;
  ctx.lineCap     = 'round';
  ctx.lineJoin    = 'round';
  let painting = false, lx = 0, ly = 0;

  cv.addEventListener('mousedown', e => {
    painting = true;
    const r = cv.getBoundingClientRect();
    lx = e.clientX - r.left;
    ly = e.clientY - r.top;
  });
  cv.addEventListener('mousemove', e => {
    if (!painting) return;
    const r  = cv.getBoundingClientRect();
    const cx = e.clientX - r.left;
    const cy = e.clientY - r.top;
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(cx, cy);
    ctx.stroke();
    lx = cx; ly = cy;
  });
  cv.addEventListener('mouseup',    () => painting = false);
  cv.addEventListener('mouseleave', () => painting = false);

  window.clearDC = () => {
    ctx.clearRect(0, 0, cv.width, cv.height);
    document.getElementById('msg').innerText = 'Cleared! Draw again.';
  };
  window.sendDC = () => {
    document.getElementById('msg').innerText = 'Predicting...';
    google.colab.kernel.invokeFunction('predict_canvas', [cv.toDataURL('image/png')], {});
  };
</script>
"""))

display(out)

print("─" * 40)
print("📁 OR upload an image below:")
display(widgets.VBox([upload_btn, upload_pred_btn, out]))
