// ArSL-BLIP Web Demo Script

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewBox = document.getElementById('preview-box');
const previewImg = document.getElementById('preview-img');
const predictBtn = document.getElementById('predict-btn');
const attentionToggle = document.getElementById('attention-toggle');
const resultSection = document.getElementById('result-section');
const attentionWrap = document.getElementById('attention-wrap');
const attentionImg = document.getElementById('attention-img');

let selectedFile = null;

// Drop zone events
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewBox.classList.remove('hidden');
    predictBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

// Predict
predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  predictBtn.textContent = 'Predicting…';
  predictBtn.disabled = true;
  resultSection.classList.add('hidden');

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('attention', attentionToggle.checked ? 'true' : 'false');

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    document.getElementById('res-letter').textContent = data.letter || '?';
    document.getElementById('res-arabic').textContent = data.arabic_char || '?';
    document.getElementById('res-confidence').textContent = (data.confidence * 100).toFixed(1) + '%';

    const statusEl = document.getElementById('res-status');
    if (data.accepted) {
      statusEl.textContent = 'ACCEPTED';
      statusEl.className = 'result-value status-badge status-accepted';
    } else {
      statusEl.textContent = 'LOW CONFIDENCE';
      statusEl.className = 'result-value status-badge status-rejected';
    }

    // Confidence bar
    const bar = document.getElementById('confidence-bar');
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = (data.confidence * 100) + '%'; }, 50);

    // Attention
    if (data.attention_image) {
      attentionImg.src = 'data:image/png;base64,' + data.attention_image;
      attentionWrap.classList.remove('hidden');
    } else {
      attentionWrap.classList.add('hidden');
    }

    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    alert('Network error: ' + err.message);
  } finally {
    predictBtn.textContent = 'Predict';
    predictBtn.disabled = false;
  }
});
