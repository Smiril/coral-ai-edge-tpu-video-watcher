#!/usr/bin/env python3.10
import argparse
import collections
import operator
import os
import time
import cv2
import numpy as np

# Optionale Imports sauber abfangen, damit das Skript auf Mac UND Linux/EdgeTPU läuft
try:
  import coremltools as ct
except ImportError:
  ct = None

try:
  import tflite_runtime.interpreter as tflite
except ImportError:
  import tensorflow.lite as tflite

try:
  from edgetpu.utils import image_processing
except ImportError:
  image_processing = None

Category = collections.namedtuple('Category', ['id', 'score'])


class CoreMLInterpreterAdapter:
  """Simuliert die tflite.Interpreter-API für Core ML (.mlpackage / .mlmodel)

  auf macOS (MPS & NPU).
  """

  def __init__(self, model_path, compute_units):
    if ct is None:
      raise ImportError('coremltools ist nicht installiert (pip install coremltools).')

    coreml_path = model_path.replace('.tflite', '.mlpackage')
    self.model = ct.models.MLModel(coreml_path, compute_units=compute_units)

    spec = self.model.get_spec()
    self._input_desc = spec.description.input
    self._output_desc = spec.description.output
    self._input_name = self._input_desc.name
    self._output_name = self._output_desc.name

    # Input-Shape ermitteln und in TFLite-Format [1, H, W, C] übersetzen
    if self._input_desc.type.HasField('imageType'):
      h = self._input_desc.type.imageType.height
      w = self._input_desc.type.imageType.width
      self._shape = np.array([1, h, w, 3], dtype=np.int32)
      self._is_image_type = True
      self._channels_first = False
    else:
      raw_shape = list(self._input_desc.type.multiArrayType.shape)
      # Prüfen ob Channels-First [1, 3, H, W] -> auf [1, H, W, 3] mappen
      if len(raw_shape) == 4 and raw_shape == 3 and raw_shape != 3:
        self._shape = np.array([raw_shape, raw_shape, raw_shape, raw_shape], dtype=np.int32)
        self._channels_first = True
      else:
        self._shape = np.array(raw_shape, dtype=np.int32)
        self._channels_first = False
      self._is_image_type = False

    self._input_tensor = None
    self._output_tensor = None

  def allocate_tensors(self):
    pass

  def get_input_details(self):
    return [{
        'index': 0,
        'name': self._input_name,
        'shape': self._shape,
        'dtype': np.float32,
    }]

  def get_output_details(self):
    return [{'index': 0, 'name': self._output_name, 'dtype': np.float32}]

  def set_tensor(self, index, value):
    if self._is_image_type:
      from PIL import Image

      img_array = np.squeeze(value, axis=0).astype(np.uint8)
      self._input_tensor = Image.fromarray(img_array)
    else:
      data = value.astype(np.float32)
      if self._channels_first and data.ndim == 4:
        data = np.transpose(data, (0, 3, 1, 2))
      self._input_tensor = data

  def invoke(self):
    prediction = self.model.predict({self._input_name: self._input_tensor})
    out = np.array(prediction[self._output_name], dtype=np.float32)
    if out.ndim == 1:
      out = np.expand_dims(out, axis=0)
    self._output_tensor = out

  def get_tensor(self, index):
    return self._output_tensor

def preprocess_frame(frame, target_shape, dtype):
  # OpenCV liefert BGR -> für MobileNetV2 in RGB umwandeln
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  img = np.expand_dims(frame_rgb, axis=0)

  # Int8-quantisiertes TFLite-Modell (z.B. EdgeTPU) erwartet uint8 (0..255)
  if dtype == np.uint8:
    return img.astype(np.uint8)

  # Float32-Modell (CPU, GPU, CoreML NPU/MPS) wurde mit rescale=1./255 trainiert
  return img.astype(np.float32) / 255.0

# --- Deine einzelnen Loader-Funktionen ---
def load_cpu_interpreter(model_path):
  return tflite.Interpreter(model_path=model_path)

def load_edgetpu_interpreter(model_path):
  edgetpu_path = model_path.replace('.tflite', '_int8_edgetpu.tflite')
	edgetpu_delegate = tflite.load_delegate('libedgetpu.so.1', options={'device': 'usb0','device': 'usb1'})
  return tflite.Interpreter(model_path=edgetpu_path, experimental_delegates=[edgetpu_delegate])

def load_gpu_interpreter(model_path):
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
  return tflite.Interpreter(model_path=model_path)

def load_mps_interpreter(model_path):
  return CoreMLInterpreterAdapter(model_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

def load_npu_interpreter(model_path):
  return CoreMLInterpreterAdapter(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

def load_interpreter(model_path, device):
  loaders = {
      'cpu': load_cpu_interpreter,
      'tpu': load_edgetpu_interpreter,
      'gpu': load_gpu_interpreter,
      'mps': load_mps_interpreter,
      'npu': load_npu_interpreter,
  }
  if device not in loaders:
    raise ValueError("Invalid device. Choose either 'cpu', 'tpu', 'gpu', 'mps' or 'npu'.")
  return loaders[device](model_path)

def get_output(interpreter, top_k, score_threshold):
  """Returns no more than top_k categories with score >= score_threshold."""
  scores = interpreter.get_tensor(interpreter.get_output_details()['index'])
  categories = [
      Category(i, scores[i])
      for i in np.argpartition(scores, -top_k)[-top_k:]
      if scores[i] >= score_threshold
  ]
  return sorted(categories, key=operator.itemgetter(1), reverse=True)

def preprocess_frame(frame, target_shape, dtype):
  """Fallback für image_processing.reshape_and_rescale, falls edgetpu nicht installiert ist."""
  if image_processing is not None:
    return image_processing.reshape_and_rescale(
        np.array(frame), target_shape
    ).astype(dtype)

  img = np.expand_dims(frame, axis=0)
  if dtype == np.float32:
    return (img.astype(np.float32) / 127.5) - 1.0
  return img.astype(dtype)

def parse_args():
  parser = argparse.ArgumentParser(description='Real-time image classification', add_help=False)
  parser.add_argument('--model_path',type=str,required=True,help='Path to .tflite or .mlpackage model',)
  parser.add_argument('--label_path', type=str, required=True, help='Path to label file')
  parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
  parser.add_argument('--width', type=int, default=640, help='Width of video frame')
  parser.add_argument('--height', type=int, default=480, help='Height of video frame')
  parser.add_argument('--device',type=str,default='cpu',choices=['cpu', 'tpu', 'gpu', 'mps', 'npu'],help='Device to use',)
  parser.add_argument('--video_device',type=int,default=0,help='Index of video capture device',)
  parser.add_argument('--help',action='help',default=argparse.SUPPRESS,help='show this help message and exit',)
  return parser.parse_args()

def main():
  args = parse_args()

  # Label-Datei laden
  with open(args.label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

  # Interpreter für das gewählte Gerät laden
  interpreter = load_interpreter(args.model_path, args.device)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Kamera initialisieren
  cap = cv2.VideoCapture(args.video_device)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Frame vorbereiten
    target_w = input_details['shape']
    target_h = input_details['shape']
    resized_frame = cv2.resize(frame, (target_w, target_h))

    input_data = preprocess_frame(resized_frame, input_details['shape'][1:3], input_details['dtype'])

    # Inferenz ausführen
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    # Ergebnisse über deine get_output-Funktion abrufen
    results = get_output(interpreter, top_k=1, score_threshold=args.threshold)

    if results:
      top_result = results
      label = (
          labels[top_result.id]
          if top_result.id < len(labels)
          else f'ID {top_result.id}'
      )
      label_text = f'{label}: {top_result.score:.2f} ({args.device.upper()})'
      cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
