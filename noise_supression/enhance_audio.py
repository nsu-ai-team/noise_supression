import noise_supression.config as cfg
import numpy as np
import noise_supression.prepare_data as pp_data

import pickle
import sys
import os

from keras.models import load_model
from noise_supression.spectrogram_to_wave import recover_wav

class AudioEnhancer:
	def __init__(self):
		self.model = load_model(os.path.join(os.path.dirname(__file__), 'model/md_10000iters.h5'))
		self.scaler = pickle.load(open(os.path.join(os.path.dirname(__file__), 'model/scaler.p'), 'rb'))
		self.n_window = cfg.n_window
		self.n_overlap = cfg.n_overlap
		self.fs = cfg.sample_rate
		self.scale = True

	def enhance_audio(self, speech_dir, output_dir, n_concat=7, n_hop=3):
		(speech_audio, _) = pp_data.read_audio(speech_dir, target_fs=self.fs)

		# Extract spectrogram. 
		mixed_complx_x = pp_data.calc_sp(speech_audio, mode='complex')

		mixed_x = np.abs(mixed_complx_x)

		# Process data. 
		n_pad = int((n_concat - 1) / 2)
		mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
		mixed_x = pp_data.log_sp(mixed_x)

		# Scale data. 
		if self.scale:
			mixed_x = pp_data.scale_on_2d(mixed_x, self.scaler)

		# Cut input spectrogram to 3D segments with n_concat. 
		mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)

		# Predict. 
		pred = self.model.predict(mixed_x_3d)

		# Inverse scale. 
		if self.scale:
			pred = pp_data.inverse_scale_on_2d(pred, self.scaler)

		# Recover enhanced wav. 
		pred_sp = np.exp(pred)
		s = recover_wav(pred_sp, mixed_complx_x, self.n_overlap, np.hamming)
		s *= np.sqrt((np.hamming(self.n_window)**2).sum())		# Scaler for compensate the amplitude 
																# change after spectrogram and IFFT. 

		pp_data.write_audio(output_dir, s, self.fs)

class AudioEnhancerTIMIT:
	def __init__(self):
		self.model = load_model('model/sednn_keras_logMag_Relu2048layer1_1outFr_7inFr_dp0.2_weights.75-0.00.hdf5')
		self.scaler = pickle.load(open('model/tr_norm.pickle', 'rb'), encoding='latin1')
		self.n_window = 512
		self.n_overlap = 256
		self.fs = 16000
		self.scale = True

	def enhance_audio(self, speech_dir, output_dir, n_concat=11, n_hop=3):
		(speech_audio, _) = pp_data.read_audio(speech_dir, target_fs=self.fs)

		# Extract spectrogram. 
		mixed_complx_x = pp_data.calc_sp(speech_audio, mode='complex')

		mixed_x = np.abs(mixed_complx_x)

		# Process data. 
		n_pad = int((n_concat - 1) / 2)
		mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
		mixed_x = pp_data.log_sp(mixed_x)

		# Scale data. 
		if self.scale:
			mixed_x = pp_data.scale_on_2d(mixed_x, self.scaler)

		# Cut input spectrogram to 3D segments with n_concat. 
		mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)

		# Predict. 
		pred = self.model.predict(mixed_x_3d)

		# Inverse scale. 
		if self.scale:
			pred = pp_data.inverse_scale_on_2d(pred, self.scaler)

		# Recover enhanced wav. 
		pred_sp = np.exp(pred)
		s = recover_wav(pred_sp, mixed_complx_x, self.n_overlap, np.hamming)
		s *= np.sqrt((np.hamming(self.n_window)**2).sum())		# Scaler for compensate the amplitude 
																# change after spectrogram and IFFT. 

		pp_data.write_audio(output_dir, s, self.fs)

if __name__ == '__main__':
	ae = AudioEnhancer()
	ae.enhance_audio(sys.argv[1], sys.argv[2])