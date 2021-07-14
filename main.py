import copy
import os
import time

import crepe
import ddsp
import ddsp.training
import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import pydub

def main(file_location, output_name):
  # Helper Functions
  DEFAULT_SAMPLE_RATE = 16000
  sample_rate = DEFAULT_SAMPLE_RATE  # 16000


  print('Done!')


  # ## Monopoly, Multipoly 파일 모두 사용할 수 있도록 수정  

  # In[4]:


  flag = None

  record_or_upload = "Upload (.mp3 or .wav)"  #@param ["Record", "Upload (.mp3 or .wav)"]

  record_seconds =      20#@param {type:"number", min:1, max:10, step:1}

  def read(f, normalized=False):
      """MP3 to numpy array"""
      a = pydub.AudioSegment.from_mp3(f)
      y = np.array(a.get_array_of_samples())
      # if a.channels == 2:
      #     y = y.reshape((-1, 2))
      if normalized:
          return a.frame_rate, np.float32(y) / 2**15
      else:
          return a.frame_rate, y


  if record_or_upload == "Record":
    audio = record(seconds=record_seconds)
  else:
    # Load audio sample here (.mp3 or .wav3 file)
    # Just use the first file.
    filenames = [file_location]
    rrate, audios = read(filenames[0],normalized=True)
    # audio = note_seq.audio_io.wav_data_to_samples_pydub(wav_data=str.encode(filenames[0]),sample_rate=16000,normalize_db=None)
    audios = [audios]
    audio = audios[0]
    
    flag = 'multi' if audio.ndim==2 else 'mono'
    print(audio.shape)
    # monopoly : (352000,)  (1, 352000)
    # non-monopoly : (2, 8681472)


  #####
  if flag == 'multi':
    audio = audio[0]
  elif flag == 'mono':
    audio = audio[np.newaxis, :]

  print(audio.shape)
  print('\nExtracting audio features...')


  ## setup session
  ddsp.spectral_ops.reset_crepe()

  # Compute features.
  start_time = time.time()
  # audio_features = ddsp.training.metrics.compute_audio_features(audio)
  audio_features={'audio':np.random.uniform(-0.3,0.3,size=(254955)), 'loudness_db':np.random.uniform(-120,-26,size=(3983)), 'f0_hz':np.random.uniform(0,495.8,size=(3983)), 'f0_confidence':np.random.uniform(0,1,size=(3983))}
  audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
  audio_features_mod = None
  print('Audio features took %.1f seconds' % (time.time() - start_time))

  #####
  model = 'Flute' #@param ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone', 'pjy_guitar', 'gjy_bottle', 'pdh_bell', 'pjy_water', 'pjy_water_mel', 'gjh_birds', 'gjh_singingball_1', 'pjy_insect_1', 'pjy_insect_2', 'pjy_dolphin'] 
  #user model : 'Upload your own (checkpoint folder as .zip)', 
  MODEL = model

  def find_model_dir(dir_name):
    # Iterate through directories until model directory is found
    for root, dirs, filenames in os.walk(dir_name):
      for filename in filenames:
        if filename.endswith(".gin") and not filename.startswith("."):
          model_dir = root
          break
    return model_dir 

  if model in ('Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone'):
    model_dir = model
    gin_file = os.path.join(model_dir, 'operative_config-0.gin')

  else:
    if model == 'pjy_guitar':
      model_dir = myPath + 'pjy_guitar_my_solo_instrument'
    elif model == 'pjy_water':
      model_dir = myPath + 'pjy_water_my_solo_instrument'
    elif model == 'pdh_bell':
      model_dir = myPath + 'pdh_bell_my_solo_instrument'
    elif model == 'gjh_bottle':
      model_dir = myPath + 'gjh_bottle_my_solo_instrument'
    elif model == 'pjy_water_mel':
      model_dir = myPath + 'pjy_water_mel_my_solo_instrument'
    elif model == 'gjh_birds':
      model_dir = myPath + 'gjh_birds_my_solo_instrument'
    elif model == 'gjh_singingball_1':
      model_dir = myPath + 'gjh_singingball_1_solo_instrument'
    elif model == 'pjy_insect_1':
      model_dir = myPath + 'pjy_insect_1_solo_instrument'
    elif model == 'pjy_dolphin':
      model_dir = myPath + 'pjy_dolphin_solo_instrument'
    
    gin_file = os.path.join(model_dir, 'operative_config-0.gin')


  # Load the dataset statistics.
  DATASET_STATS = None
  dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
  print(f'Loading dataset statistics from {dataset_stats_file}')
  try:
    if tf.io.gfile.exists(dataset_stats_file):
      with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
        DATASET_STATS = pickle.load(f)
  except Exception as err:
    print('Loading dataset statistics from pickle failed: {}.'.format(err))


  # Parse gin config,
  with gin.unlock_config():
    gin.parse_config_file(gin_file, skip_unknown=True)

  # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
  ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
  ckpt_name = ckpt_files[0].split('.')[0]
  ckpt = os.path.join(model_dir, ckpt_name)

  # Ensure dimensions and sampling rates are equal
  time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
  n_samples_train = gin.query_parameter('Harmonic.n_samples')
  hop_size = int(n_samples_train / time_steps_train)

  if flag == 'mono':
    time_steps = int(audio.shape[1] / hop_size)
  elif flag == 'multi':
    time_steps = int(audio.shape[0] / hop_size)

  n_samples = time_steps * hop_size

  print("===Trained model===")
  print("Time Steps", time_steps_train)
  print("Samples", n_samples_train)
  print("Hop Size", hop_size)
  print("\n===Resynthesis===")
  print("Time Steps", time_steps)
  print("Samples", n_samples)
  print('')

  gin_params = [
      'Harmonic.n_samples = {}'.format(n_samples),
      'FilteredNoise.n_samples = {}'.format(n_samples),
      'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
      'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
  ]

  with gin.unlock_config():
    gin.parse_config(gin_params)


  # Trim all input vectors to correct lengths 
  for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
    audio_features[key] = audio_features[key][:time_steps]

  if flag == 'multi':
    audio_features['audio'] = audio_features['audio'][:, :n_samples]
  elif flag == 'mono': 
    audio_features['audio'] = audio_features['audio'][ :n_samples]


  # Set up the model just to predict audio given new conditioning
  model = ddsp.training.models.Autoencoder()
  model.restore(ckpt)
  print(audio_features)
  # Build model by running a batch through it.
  start_time = time.time()
  _ = model(audio_features, training=False)
  print('Restoring model took %.1f seconds' % (time.time() - start_time))
  # cpu : 481.2 seconds / 361.1 sec


  ############ customize
  start_time = time.time()
  audio_gen = model.get_audio_from_outputs(_)
  print('Prediction took %.1f seconds' % (time.time() - start_time))
  # cpu : 0 seconds
  print('vars(model).keys()',vars(model).keys())
  print('vars(audio_gen).keys()', vars(audio_gen).keys())
  print('type(audio_gen), type(audio) :' , type(audio_gen), type(audio))

  # write

  def write(f, sr, x, normalized=False):
      """numpy array to MP3"""
      channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
      if normalized:  # normalized array - each item should be a float in [-1, 1)
          y = np.int16(x * 2 ** 15)
      else:
          y = np.int16(x)
      song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
      song.export(f, format="mp3", bitrate="320k")
  PPath = os.path.join(output_name)
  write(PPath,16000, audio_gen, normalized=True)






if __name__=="__main__":
  filename = "202003294049_89bpm.mp3"
  output_name = 'neww.mp3'
  main(filename, output_name)