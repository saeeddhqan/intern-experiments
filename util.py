from subprocess import CalledProcessError, run
from functools import lru_cache
from typing import Optional, Union, NoReturn, Any, List, Tuple
import numpy as np
from datetime import datetime

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import math, os, pathlib, random, argparse, json, re
import sentencepiece, datasets, tqdm, hashlib, urllib


def download_whisper(url: str, root: str):
	os.makedirs(root, exist_ok=True)

	expected_sha256 = url.split("/")[-2]
	download_target = os.path.join(root, os.path.basename(url))

	if os.path.exists(download_target) and not os.path.isfile(download_target):
		raise RuntimeError(f"{download_target} exists and is not a regular file")

	if os.path.isfile(download_target):
		with open(download_target, "rb") as f:
			model_bytes = f.read()
		if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
			return download_target
		else:
			print(
				f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
			)

	with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
		with tqdm.tqdm(
			total=int(source.info().get("Content-Length")),
			ncols=80,
			unit="iB",
			unit_scale=True,
			unit_divisor=1024,
		) as loop:
			while True:
				buffer = source.read(8192)
				if not buffer:
					break

				output.write(buffer)
				loop.update(len(buffer))

	model_bytes = open(download_target, "rb").read()
	if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
		raise RuntimeError(
			"Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
		)
	return download_target




def get_logger(log_dir: str, name: str, log_filename: str = 'log', require_writer: bool = True):
	os.makedirs(log_dir, exist_ok=True)

	class CustomLogger:
		def __init__(self, name):
			self.name = name
			if require_writer:
				self.log_file = open(os.path.join(log_dir, log_filename + '.log'), 'a')

		def info(self, message):
			log_entry = f"{datetime.now()} - {self.name} - INFO - {message}"
			if require_writer:
				self.log_file.write(log_entry + '\n')
			print(log_entry)

		def __del__(self):
			if require_writer:
				self.log_file.close()

	print('Log directory: ', log_dir)
	return CustomLogger(name)


class TextProcess:
	def __init__(self, tokenizer_model_path: str = '', digit_dataset: bool = True) -> NoReturn:
		self.sot = '<s>'
		self.eot = '</s>'
		self.pad = '<pad>'
		if digit_dataset:
			chars = [self.sot, self.eot, self.pad]
			self.chars = chars + sorted(list(set('0123456789')))
			self.stoi = {x:i for i,x in enumerate(self.chars)}
			self.itos = {i:x for i,x in enumerate(self.chars)}
			self.encode = lambda s: [self.stoi[x] for x in s]
			self.decode = lambda e: ''.join([self.itos[x] for x in e])
			self.sot_i = self.stoi[self.sot]
			self.eot_i = self.stoi[self.eot]
			self.vocab_size = len(self.chars)

			self.sot_i = 0
			self.eot_i = 1
			self.pad_i = 2
		else:
			self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_model_path)
			self.encode = self.tokenizer.encode
			self.decode = lambda seq: self.tokenizer.decode(seq)
			self.vocab_size = self.tokenizer.vocab_size()
			self.sot_i = 1
			self.eot_i = 2
			self.pad_i = 3

		self.clean = lambda s: s.replace(self.sot, '').replace(self.eot, '')


	def __len__(self) -> int:
		return self.vocab_size


	def encoder(self, text: str, block_size: int) -> List:
		text = text.replace('\n', '')
		text = [x for x in self.encode(text)]
		text.insert(self.sot_i, 0)

		if len(text) < block_size:
			text.append(self.eot_i)
			text.extend([self.pad_i for _ in range(block_size - len(text))])
			return text
		elif len(text) >= block_size:
			print(f'[{len(text)}]cutting...')
			text = text[:block_size - 1]
		text.append(self.eot_i)
		return text


	def decoder(self, s: Union[Tensor, list], remove_special_chars: Optional[bool] = False) -> str:
		'''
			Parameters
			----------
			s: str
				The string to decode
			remove_special_chars: bool
				Whether to remove the start and end of text tokens
		'''
		return self.clean(self.decode(s))


def wer(r: str, h: str):
	'''
		Calculate word error rate (WER), the lower the better.
		Parameters
		----------
		r: str
			The ground truth text transcript
		h: str
			The predicted text transcript
		Returns
		-------
		float:
			The word error rate
	'''
	# build the matrix
	d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
	d = d.reshape((len(r) + 1, len(h) + 1))
	for i in range(len(r) + 1):
		for j in range(len(h) + 1):
			if i == 0:
				d[0][j] = j
			elif j == 0:
				d[i][0] = i
	
	# computation
	for i in range(1, len(r) + 1):
		for j in range(1, len(h) + 1):
			if r[i - 1] == h[j - 1]:
				d[i][j] = d[i - 1][j - 1]
			else:
				substitution = d[i - 1][j - 1] + 1
				insertion = d[i][j - 1] + 1
				deletion = d[i - 1][j] + 1
				d[i][j] = min(substitution, insertion, deletion)
	
	return d[len(r)][len(h)] / len(r)


def accuracy(r: str, h: str):
	'''
		Calculate accuracy, the higher the better.
		Parameters
		----------
		r: str
			The ground truth text transcript
		h: str
			The predicted text transcript
		Returns
		-------
		float:
			The accuracy
	'''
	assert len(r) == len(h)
	return float(sum([1 for x, y in zip(r, h) if x == y])) / len(r)


def acc_wer_score(r: str, h: str):
	'''
		Calculate the overall performance, the higher the better.
	'''
	return accuracy(r, h) * (1 - wer(r, h))


def overall_accuracy(all_r: list, all_h: list):
	'''
		metrics
	'''
	assert len(all_r) == len(all_h), f"Lengths of r and h should be the same. Got {len(all_r)} and {len(all_h)}"
	score_acc = 0
	score_acc_wer = 0
	score_wer = 0
	for r, h in zip(all_r, all_h):
		score_acc += accuracy(r, h)
		score_acc_wer += acc_wer_score(r, h)
		score_wer += wer(r, h)
	score_acc = score_acc / len(all_r)
	score_acc_wer = score_acc_wer / len(all_r)
	score_wer_raw = score_wer / len(all_r)
	score_wer = (1 - score_wer_raw) # Now, the higher the better
	one_score_for_all = (score_acc + score_acc_wer + score_wer) / 3
	return score_acc, score_acc_wer, score_wer, one_score_for_all, score_wer_raw


def plot_metrics(metrics_list, train_id):
	"""
	Plot train and test loss, word error rate, and accuracy.
	"""
	key_plot = 'micro' if config.epoch < 50 and len(metrics_list['micro']) > 0 else 'main'
	order = ('main', 'micro') if key_plot == 'micro' else ('micro', 'main')
	best_metrics = None
	min_wer = float('inf')
	content = []

	for key in order:
		epochs = [t[0] for t in metrics_list[key]]
		steps = [t[1] for t in metrics_list[key]]
		train_losses = [t[2].item() for t in metrics_list[key]]
		test_losses = [t[3].item() for t in metrics_list[key]]
		wer = [t[4] for t in metrics_list[key]]
		accuracy = [t[5] for t in metrics_list[key]]

		content.append({
			f"train_losses_{key}": train_losses,
			f"test_losses_{key}": test_losses,
			f"wer_{key}": wer,
			f"accuracy_{key}": accuracy,
		})
		if wer == []:
			continue
		if min(wer) < min_wer:
			min_wer = min(wer)
			min_index = wer.index(min_wer)
			best_metrics = (accuracy[min_index], min_wer, train_losses[min_index], test_losses[min_index],)

	json.dump(content, open(f"logs/{train_id}.json", 'w'))

	config.logger.info(f"Using {key_plot} for plots")

	combined_epochs = np.array(epochs) + np.array(steps) / max(steps) if key_plot == 'micro' else epochs

	plt.plot(combined_epochs, train_losses, label='Train Loss')
	plt.plot(combined_epochs, test_losses, label='Test Loss')
	plt.title('Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()

	plt.savefig(f"logs/{train_id}_loss.png")
	plt.clf()

	plt.plot(combined_epochs, wer)
	plt.title('Word Error Rate')
	plt.xlabel('Epoch')
	plt.ylabel('WER')
	plt.savefig(f"logs/{train_id}_wer.png")
	plt.clf()

	plt.plot(combined_epochs, accuracy)
	plt.title('Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.savefig(f"logs/{train_id}_accuracy.png")
	plt.clf()

	return best_metrics 



class Config:
	def __init__(self, data_dict: dict) -> NoReturn:
		'''
			Given a data_dict, the class treats each key/val as an object.
			Parameters
			----------
			data_dict: dict
				a dict that key is a property and value is its value
		'''
		self.__data_dict__ = data_dict

	def __getattr__(self, k: Union[int, str, bytes]) -> Any:
		'''
			Given a key, it returns its data if it exists, otherwise None.
			Parameters
			----------
			k: str
				key
			Returns
			-------
			v: Union[any type]
				the value of the k
		'''
		if k in self.__data_dict__:
			return self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")


	def __setattr__(self, k: Union[int, str, bytes], v: Any) -> NoReturn:
		'''
			Given a key, and value, it sets the key with the corresponding value.
		'''
		if k == '__data_dict__':
			super().__setattr__(k, v)
		else:
			self.__data_dict__[k] = v


	def __delattr__(self, k: Union[int, str, bytes]) -> NoReturn:
		'''
			Given a key, it deletes it from data dict if it exists.
		'''
		if k in self.__data_dict__:
			del self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")


	def set_args(self, args: argparse.Namespace) -> NoReturn:
		'''
			Given an object of argparse, the method adds all the KVs to the data.
		'''
		for kv in args._get_kwargs():
			k, v = kv
			self.__setattr__(k, v)


	def get_model_params(self, abstract: bool = False) -> dict:
		'''
			Returns a dictionary that contains model parameters.
		'''
		filters = ('data_load', 'load', 'iterations', 'autocast')
		params = {}
		for k in self.__data_dict__:
			if k not in filters:
				params[k] = self.__data_dict__[k]
		return params


	def set_model_params(self, params: dict) -> NoReturn:
		'''
			Returns a dictionary that contains model specifications.
		'''

		filters = (
			'data_load', 'action', 'load', 'workdir', 'mode')
		for k in params:
			if k not in filters:
				self.__data_dict__[k] = params[k]



whisper_models = {
	"tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
	"small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
	"small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
}

SAMPLE_RATE = 16000
N_FFT = 400 # win length
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30 # supported duration (per second)
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 10ms per audio frame
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 20ms per audio token

use_dataset = 'boolq'
if use_dataset == 'boolq':
	text_process = TextProcess(tokenizer_model_path='assets/boolq/boolq-tok-8k.model', digit_dataset=False)
else:
	text_process = TextProcess(digit_dataset=True)

exact_div = lambda a, b: a // b
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_VOCABS = len(text_process)
seqlen = 32

params = {
	'model_name': 'Whisper',
	'device': device,
	'text_process': text_process,
	'train_path': os.path.join(os.path.dirname(__file__), 'train_data'),
	'test_path': os.path.join(os.path.dirname(__file__), 'test_data'),
	'noise_path': os.path.join(os.path.dirname(__file__), 'noise_dir'),
	'checkpoint_dir': 'checkpoints',
	'log_dir': 'logs',
	'epoch': 500,
	'test_steps': 40,
	'model_mode': 'train',
	'test_freq': 5,
	'batch_size': 16,
	'seq_len': seqlen,
	'n_vocab': N_VOCABS,
	'specaug_rate': 0.3,
	'freq_mask': 27,
	'time_mask': 70,
	'sample_rate': SAMPLE_RATE, 
	'n_mels': N_MELS,
	'n_fft': N_FFT, 
	'hop_length': HOP_LENGTH,
	'chunk_length': CHUNK_LENGTH,
	'n_samples': N_SAMPLES,
	'n_frames': N_FRAMES,
	'accumulation_steps': 1,
	'dtype': torch.float32,
	'dim': 64,
	'nlayers': 4,
	'nheads': 2,

	'use_noise_background': True,
	'use_speed_change': True,
	'use_freq_mask': False,
	'use_time_stretch': False,
	'noise_background_dir': 'noise_dir',
	'regularization_on_raw_audio': True,
	'regularization_on_mel': False,
	'regularization_on_data': True,

	'lr': 1e-3,
	'n_audio_ctx': (CHUNK_LENGTH * 100) // 2,

	'n_text_ctx': seqlen,

	'audio_dropout': 0.1,
	'text_dropout': 0.1,
	'attention_dropout': 0.1,
	'wandb': False,
	'causal_mode': 'non-causal', # causal, non-causal, semi-causal, bw-semi-causal
	'variation': '',
	'logger': None,
	'log_anything': True,
	'save_checkpoints': False,
	'model_path': '',
	'train_id': '',
	'use_dataset': use_dataset,
	'one_shot': False,
	'fine_tune': False,
	'no_footprint': False,
	'freeze_encoder': False,
}

config = Config(params)
config.autocast = torch.autocast(device_type=config.device, dtype=config.dtype)
cache = {}


class RandomBackgroundNoise(nn.Module):
	def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15):
		super(RandomBackgroundNoise, self).__init__()
		self.sample_rate = sample_rate
		self.min_snr_db = min_snr_db
		self.max_snr_db = max_snr_db

		if not os.path.exists(noise_dir):
			raise IOError(f'Noise directory `{noise_dir}` does not exist')
		# find all WAV files including in sub-folders:
		self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
		if len(self.noise_files_list) == 0:
			raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

	def __call__(self, audio_data):
		random_noise_file = random.choice(self.noise_files_list)
		effects = [
			['remix', '1'], # convert to mono
			['rate', str(self.sample_rate)], # resample
		]
		noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
		audio_length = audio_data.shape[-1]
		noise_length = noise.shape[-1]
		if noise_length > audio_length:
			offset = random.randint(0, noise_length-audio_length)
			noise = noise[..., offset:offset+audio_length]
		elif noise_length < audio_length:
			noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim=-1)

		snr_db = random.randint(self.min_snr_db, self.max_snr_db)
		snr = math.exp(snr_db / 10)
		audio_power = audio_data.norm(p=2)
		noise_power = noise.norm(p=2)
		scale = snr * noise_power / audio_power

		return (scale * audio_data + noise ) / 2


class RandomSpeedChange(nn.Module):
	def __init__(self, sample_rate):
		super(RandomSpeedChange, self).__init__()
		self.sample_rate = sample_rate

	def forward(self, audio_data):
		'''
			Change the speed of the audio data by a random factor
			Parameters:
			-----------
			audio_data: Tensor
				The audio data to change the speed of
			Returns:
			--------
			Tensor
				The audio data with the speed changed
		'''
		speed_factor = random.choice([0.9, 1.0, 1.1, 1.2])
		if speed_factor == 1.0: # no change
			return audio_data

		# change speed and resample to original rate:
		sox_effects = [
			["speed", str(speed_factor)],
			["rate", str(self.sample_rate)],
		]
		transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
			audio_data, self.sample_rate, sox_effects)
		return transformed_audio
 

class TimeStretch(nn.Module):
	def __init__(self):
		super(TimeStretch, self).__init__()
		pass
	
	def forward(self, x: Tensor):
		'''
			Apply time stretch to the input tensor
			Parameters:
			-----------
			x: Tensor
				The input tensor
			Returns:
			--------
			Tensor
				The input tensor with time stretch applied
		'''
		factor = torch.rand(1).item() * 0.4 + 0.8
		y = torchaudio.functional.phase_vocoder(x, rate=factor, phase_advance=0.0)
		x = y.to(x.dtype)
		return x


class Augmentator(nn.Module):

	def __init__(self, rate: Optional[int] = config.specaug_rate,
			freq_mask: Optional[int] = config.freq_mask,
			time_mask: Optional[int] = config.time_mask,
		):
		super(Augmentator, self).__init__()
		self.rate = rate
		self.before = config.regularization_on_raw_audio
		self.after = config.regularization_on_mel
		# Regularization on raw audio
		self.noise = RandomBackgroundNoise(config.sample_rate, config.noise_path)
		self.speed = RandomSpeedChange(config.sample_rate)
		# Regularization on mel
		self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask)
		self.time_stretch = TimeStretch()


	def forward(self, x: Tensor, before_or_after: str):
		'''
			Apply augmentation to the input tensor
			Parameters:
			-----------
			x: Tensor
				The input tensor
			before_or_after: str
				Whether to apply the augmentation before or after the spectrogram
			Returns:
			--------
			Tensor
				The input tensor with Augmentation applied
		'''
		prob = torch.rand(1).item()
		if prob > self.rate:
			return x

		if before_or_after == 'before' and self.before:
			if config.use_noise_background:
				x = self.noise(x)
			if config.use_speed_change:
				x = self.speed(x)
		elif before_or_after == 'after' and self.after:
			if config.use_freq_mask:
				x = self.freq_mask(x)
			if config.use_time_stretch:
				x = self.time_stretch(x)
		return x

	def before(self, x: Tensor):
		'''
			Apply augmentation to the input tensor that is raw audio
			Parameters:
			-----------
			x: Tensor
				The input tensor
			Returns:
			--------
			Tensor
				The input tensor with SpecAugment applied
		'''
		probability = torch.rand(1).item()
		if probability > self.rate:
			x = self.augment_before(x)
		return x

	def after(self, x: Tensor):
		'''
			Apply augmentation to the input tensor that is a spectrogram
			Parameters:
			-----------
			x: Tensor
				The input tensor
			Returns:
			--------
			Tensor
				The input tensor with SpecAugment applied
		'''
		probability = torch.rand(1).item()
		if probability > self.rate:
			x = self.augment_after(x)
		return x


def load_audio(file: str, sr: int = config.sample_rate):
	'''
		Open an audio file and read as mono waveform, resampling as necessary

		Parameters
		----------
		file: str
			The audio file to open

		sr: int
			The sample rate to resample the audio if necessary

		Returns
		-------
		A NumPy array containing the audio waveform, in float32 dtype.
	'''

	# This launches a subprocess to decode audio while down-mixing
	# and resampling as necessary.  Requires the ffmpeg CLI in PATH.
	# fmt: off
	cmd = [
		'ffmpeg',
		'-nostdin',
		'-threads', '0',
		'-i', file,
		'-f', 's16le',
		'-ac', '1',
		'-acodec', 'pcm_s16le',
		'-ar', str(sr),
		'-'
	]
	# fmt: on
	try:
		out = run(cmd, capture_output=True, check=True).stdout
	except CalledProcessError as e:
		raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

	return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = config.n_samples, *, axis: int = -1):
	'''
		Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
	'''
	if torch.is_tensor(array):
		if array.shape[axis] > length:
			array = array.index_select(
				dim=axis, index=torch.arange(length, device=array.device)
			)

		if array.shape[axis] < length:
			pad_widths = [(0, 0)] * array.ndim
			pad_widths[axis] = (0, length - array.shape[axis])
			array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
	else:
		if array.shape[axis] > length:
			array = array.take(indices=range(length), axis=axis)

		if array.shape[axis] < length:
			pad_widths = [(0, 0)] * array.ndim
			pad_widths[axis] = (0, length - array.shape[axis])
			array = np.pad(array, pad_widths)

	return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = config.n_mels) -> torch.Tensor:
	'''
		load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
		Allows decoupling librosa dependency; saved using:

			np.savez_compressed(
				"mel_filters.npz",
				mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
			)
	'''

	assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
	with np.load(
		os.path.join(os.path.dirname(__file__), 'assets', 'mel_filters.npz')
	) as f:
		return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def apply_spec_augment(spectrogram, num_time_masks=10, pS=0.05, F=27, W=20):
	spectrogram = spectrogram.mT.contiguous()
	T, F_bins = spectrogram.shape

	# Time Warping
	def time_warping(spectrogram, W=20):
		spectrogram = spectrogram.cpu().numpy()
		center = T // 2
		warped = np.copy(spectrogram)
		point_to_warp = np.random.randint(center - W, center + W)
		dist = np.random.randint(-W, W)
		src = np.arange(T)
		dst = np.copy(src)
		dst[point_to_warp:] += dist
		for f in range(F_bins):
			warped[:, f] = np.interp(src, dst, spectrogram[:, f])
		return warped

	augmented_spectrogram = torch.from_numpy(time_warping(spectrogram, W=W)).to(spectrogram.device)

	# Frequency Masking
	for _ in range(np.random.randint(1, F+1)):
		f = np.random.randint(0, F + 1)
		f_start = np.random.randint(0, max(0, F_bins - f + 1))
		augmented_spectrogram[:, f_start:f_start + f] = 0

	# Time Masking
	max_time_mask = int(np.ceil(pS * T))
	for _ in range(num_time_masks):
		t = np.random.randint(0, max_time_mask + 1)
		t_start = np.random.randint(0, max(0, T - t + 1))
		augmented_spectrogram[t_start:t_start + t, :] = 0
	
	return augmented_spectrogram.mT.contiguous()


def log_mel_spectrogram(
	audio: Union[str, np.ndarray, Tensor],
	n_mels: int = config.n_mels,
	padding: int = 0,
	device: Optional[Union[str, torch.device]] = None,
	augmentator: Optional[Augmentator] = None,
):
	'''
		Compute the log-Mel spectrogram of

		Parameters
		----------
		audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
			The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

		n_mels: int
			The number of Mel-frequency filters, only 80 is supported

		padding: int
			Number of zero samples to pad to the right

		device: Optional[Union[str, torch.device]]
			If given, the audio tensor is moved to this device before STFT

		Returns
		-------
		torch.Tensor, shape = (80, n_frames)
			A Tensor that contains the Mel spectrogram
	'''
	if isinstance(audio, str):
		audio_path = audio
		if audio_path not in cache:
			audio = load_audio(audio)
			audio = torch.from_numpy(audio)
			cache[audio_path] = audio
		else:
			audio = cache[audio_path]
	if isinstance(audio, np.ndarray):
		audio = torch.from_numpy(audio)

	if augmentator is not None:
		audio = augmentator(audio.view(1, -1), before_or_after='before').view(-1)

	if device is not None:
		audio = audio.to(device)

	if padding > 0:
		audio = F.pad(audio, (0, padding))

	window = torch.hann_window(config.n_fft).to(audio.device)
	stft = torch.stft(audio, config.n_fft, config.hop_length, window=window, return_complex=True)
	magnitudes = stft[..., :-1].abs() ** 2

	filters = mel_filters(audio.device, n_mels)
	mel_spec = filters @ magnitudes

	log_spec = torch.clamp(mel_spec, min=1e-10).log10()
	log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
	log_spec = (log_spec + 4.0) / 4.0
	return log_spec


def prepare_audio(
	audio: Union[str, np.ndarray, Tensor],
	device: Union[str, torch.device],
	augmentator: Optional[Union[None, Augmentator]] = None,
):
	'''
		Prepare the audio file for training.
		Parameters
		----------
		audio:
			The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz
		device:
			Device name
		augmentator:
			Augmentator class if any
		Returns
		-------
		mel_segment: torch.Tensor, shape = (80, n_frames)
			The log-Mel spectrogram of the audio segment
	'''

	mel = log_mel_spectrogram(audio, padding=config.n_samples, device=device, augmentator=augmentator)
	mel = pad_or_trim(mel, config.n_frames)

	# if augmentator:
	# 	mel = apply_spec_augment(mel)

	return mel


def prepare_text(text: str, device: Union[str, torch.device]):
	'''
		Prepare the text transcript for training.
		Parameters
		----------
		text: str
			The text transcript
		Returns
		-------
		sequence: torch.Tensor, shape = (n_text_ctx,)
			The encoded text sequence

		labels: torch.Tensor, shape = (n_text_ctx,)
	'''

	sequence = [config.text_process.sot_i] + config.text_process.encoder(text, config.seq_len - 1) # one for sot
	labels = sequence[1:] + [config.text_process.eot_i]
	sequence = torch.tensor(sequence).to(device)
	labels = torch.tensor(labels).to(device)
	return sequence, labels


class DataDigit(torch.utils.data.Dataset):
	'''
		Load the data from the json file.
		The json file should be a list of objects with the following keys:
			key: path to the audio file
			text: the text transcript
	'''

	def __init__(self, dir_path: Tensor, device: Union[str, torch.device]):
		self.device = device
		self.text_process = config.text_process
		self.data = []
		self.mode = dir_path.split('/')[-1]

		self.augmentator = None
		if self.mode == 'train_data':
			self.augmentator = Augmentator()

		file_list = os.listdir(dir_path)
		for filename in file_list:
			_, extension = os.path.splitext(filename)
			if extension == '.txt':
				continue
			file_path_voice = os.path.join(dir_path, filename)
			file_path_text = os.path.join(dir_path, filename.replace(extension, '.txt'))
			if not (os.path.isfile(file_path_voice) and os.path.isfile(file_path_voice)):  # Check if it's a file (not a subdirectory)
				continue

			self.data.append({'key': file_path_voice, 'text': file_path_text})


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx: int):
		'''
			Returns
			-------
			mel_segment: torch.Tensor, shape = (80, n_frames)
				The log-Mel spectrogram of the audio segment

			sequence: torch.Tensor, shape = (n_text_ctx,)
				The encoded text sequence

			labels: torch.Tensor, shape = (n_text_ctx,)
		'''
		if torch.is_tensor(idx):
			idx = idx.item()

		file_path = self.data[idx]['key']
		mel_segment = prepare_audio(file_path, self.device, self.augmentator)
		text = open(self.data[idx]['text']).read()
		sequence, labels = prepare_text(text, self.device)
		return mel_segment, sequence, labels


class DataBoolq(torch.utils.data.Dataset):

	def __init__(self, mode: str, device: Union[str, torch.device]):
		self.device = device
		self.text_process = config.text_process
		self.data = []
		self.mode = mode
		ds = datasets.load_dataset('fixie-ai/boolq-audio')


		self.augmentator = None
		if self.mode == 'train_data':
			self.augmentator = Augmentator()
			self.data = ds['train']
		else:
			self.data = ds['validation']
		self.data = self.data.remove_columns('answer')
		self.data = self.data.remove_columns('passage')
		self.data = self.data.remove_columns('explanation')


	def normalizer(self, txt):
		txt = re.sub(r'[\?,\.\:/\]\[\{\}\=\+\(\)\!\$\%\&\*\'\"]+', '', txt)
		return txt.lower()


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx: int):
		'''
			Returns
			-------
			mel_segment: torch.Tensor, shape = (80, n_frames)
				The log-Mel spectrogram of the audio segment

			sequence: torch.Tensor, shape = (n_text_ctx,)
				The encoded text sequence

			labels: torch.Tensor, shape = (n_text_ctx,)
		'''
		if torch.is_tensor(idx):
			idx = idx.item()

		text = self.normalizer(self.data[idx]['question'])
		mel_segment = prepare_audio(np.float32(self.data[idx]['audio']['array']), self.device, self.augmentator)
		sequence, labels = prepare_text(text, self.device)
		return mel_segment, sequence, labels
