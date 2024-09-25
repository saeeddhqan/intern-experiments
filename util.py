from subprocess import CalledProcessError, run
from functools import lru_cache
from typing import Optional, Union, NoReturn, Any, List, Tuple, ClassVar, Dict
import numpy as np
from datetime import datetime

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import math, os, pathlib, random, argparse, json, re
import sentencepiece, datasets, tqdm, hashlib, urllib
import tokenizer

from torchmetrics.text import WordErrorRate, CharErrorRate
calculate_wer = WordErrorRate()
calculate_cer = CharErrorRate()


class Config:
	def __init__(self, data_dict: dict) -> NoReturn:
		self.__data_dict__ = data_dict


	def __getattr__(self, k: Union[int, str, bytes]) -> Any:
		if k in self.__data_dict__:
			return self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")


	def __setattr__(self, k: Union[int, str, bytes], v: Any) -> NoReturn:
		if k == '__data_dict__':
			super().__setattr__(k, v)
		else:
			self.__data_dict__[k] = v


	def __delattr__(self, k: Union[int, str, bytes]) -> NoReturn:
		if k in self.__data_dict__:
			del self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")


	def set_args(self, args: argparse.Namespace) -> NoReturn:
		for kv in args._get_kwargs():
			k, v = kv
			self.__setattr__(k, v)


	def get_model_params(self, abstract: bool = False) -> dict:
		filters = ('data_load', 'load', 'iterations', 'autocast')
		params = {}
		for k in self.__data_dict__:
			if k not in filters:
				params[k] = self.__data_dict__[k]
		return params


	def set_model_params(self, params: dict) -> NoReturn:
		filters = (
			'data_load', 'action', 'load', 'workdir', 'mode')
		for k in params:
			if k not in filters:
				self.__data_dict__[k] = params[k]


class TextProcess:
	def __init__(self, tokenizer_model_path: str = '', tokenizer_type: bool = True) -> NoReturn:
		self.sot = '<s>'
		self.eot = '</s>'
		self.pad = '<pad>'
		self.etc = ''
		self.tokenizer_type = tokenizer_type
		if tokenizer_type == 'digits':
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
		elif tokenizer_type == 'whisper':
			self.tokenizer = tokenizer.get_tokenizer(False, num_languages=99, language='en', task='transcribe')
			self.encode = self.tokenizer.encode
			self.decode = self.tokenizer.decode
			self.vocab_size = self.tokenizer.encoding.n_vocab
			self.sot_i = list(self.tokenizer.sot_sequence)
			self.eot_i = self.tokenizer.eot
			self.pad_i = self.tokenizer.no_speech
			self.sot = '<|startoftranscript|>'
			self.eot = '<|endoftext|>'
			self.pad = '<|nospeech|>'
			self.etc = '<|notimestamps|>'
		else:
			self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_model_path)
			self.encode = self.tokenizer.encode
			self.decode = lambda seq: self.tokenizer.decode(seq)
			self.vocab_size = self.tokenizer.vocab_size()
			self.sot_i = 1
			self.eot_i = 2
			self.pad_i = 3

		self.clean = lambda s: s.replace(self.sot, '').replace(self.eot, '').replace(self.pad, '').replace(self.etc, '')

	def __len__(self) -> int:
		return self.vocab_size


	def encoder(self, text: str, block_size: int) -> List:
		'''
			Parameters
			----------
			text: str
				The string for encoding
			block_size: int
				Set block size to fit the sequence length with it
		'''
		text = text.replace('\n', '')
		text = [x for x in self.encode(text)]
		if self.tokenizer_type == 'whisper':
			text = self.sot_i + text
		else:
			text.insert(self.sot_i, 0)

		if len(text) < block_size:
			text.extend([self.pad_i for _ in range((block_size - len(text)) - 1)])
			text.append(self.eot_i)
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


def download_whisper(url: str, root: str) -> str:
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
			total=int(source.info().get('Content-Length')),
			ncols=80,
			unit='iB',
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


def get_logger(log_dir: str, name: str,
	log_filename: str = 'log',
	require_writer: bool = True) -> ClassVar:

	os.makedirs(log_dir, exist_ok=True)

	class CustomLogger:
		def __init__(self, name: str) -> NoReturn:
			self.name = name
			if require_writer:
				self.log_file = open(os.path.join(log_dir, log_filename + '.log'), 'a')

		def info(self, message: str) -> NoReturn:
			log_entry = f"{datetime.now()} - {self.name} - INFO - {message}"
			if require_writer:
				self.log_file.write(log_entry + '\n')
			print(log_entry)

		def __del__(self) -> NoReturn:
			if require_writer:
				self.log_file.close()

	print('Log directory: ', log_dir)
	return CustomLogger(name)


def plot_metrics(metrics_list: Dict, train_id: str) -> Tuple:
	"""
		Plot train and test loss, word error rate, and accuracy.
		And find the best performing model.
		Parameters
		----------
		metrics_list: dict
			A list of captured metrics
		train_id: str
			Train id
	"""

	key_plot = 'micro' if config.epoch < 50 and len(metrics_list['micro']) > 0 else 'main'
	best_metrics = (-1, -1, -1)
	if metrics_list['instances'] == []:
		return best_metrics
	wer = [t[4] for t in metrics_list['instances']]
	min_wer = min(wer)

	train_losses = []
	test_losses = []
	wers = []
	for t in metrics_list['instances']:
		if t[4] == min_wer:
			best_metrics = (min_wer, t[2].item(), t[3].item())
		train_losses.append(t[2].item())
		test_losses.append(t[3].item())
		wers.append(t[4])


	json.dump([train_losses, test_losses, wers], open(f"logs/{train_id}.json", 'w'))

	combined_steps = np.arange(len(train_losses))

	plt.plot(combined_steps, train_losses, label='Train Loss')
	plt.plot(combined_steps, test_losses, label='Test Loss')
	plt.title('Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()

	plt.savefig(f"logs/{train_id}_loss.png")
	plt.clf()

	plt.plot(combined_steps, wers)
	plt.title('Word Error Rate')
	plt.xlabel('Epoch')
	plt.ylabel('WER')
	plt.savefig(f"logs/{train_id}_wer.png")
	plt.clf()

	return best_metrics


def set_dataset_specifiers() -> NoReturn:
	if config.dataset_name == 'digits':
		text_process = TextProcess(tokenizer_type='digits')
		CHUNK_LENGTH = 7
		seqlen = 7
	elif config.dataset_name == 'boolq':
		text_process = TextProcess(tokenizer_model_path='assets/boolq/boolq-tok-8k.model', tokenizer_type='sp')
		CHUNK_LENGTH = 30
		seqlen = 32
	else:
		text_process = TextProcess(tokenizer_type='whisper')
		CHUNK_LENGTH = 30
		seqlen = 128 # 448

	SAMPLE_RATE = 16000
	N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
	HOP_LENGTH = 160
	N_FRAMES = N_SAMPLES // HOP_LENGTH

	# not used in code, but helps
	N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
	FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 10ms per audio frame
	TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 20ms per audio token
	config.seqlen = seqlen
	config.text_process = text_process
	config.n_vocab = len(text_process)
	config.chunk_length = CHUNK_LENGTH
	config.n_samples = N_SAMPLES
	config.n_frames = N_FRAMES
	config.n_text_ctx = seqlen
	config.n_audio_ctx = (CHUNK_LENGTH * 100) // 2


whisper_models = {
	'tiny.en': 'https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt',
	'small.en': 'https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt',
	'small': 'https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt',
}

params = {
	'model_name': 'Whisper',
	'device': 'cuda' if torch.cuda.is_available() else 'cpu',
	'text_process': None,
	'train_path': os.path.join(os.path.dirname(__file__), 'train_data'),
	'test_path': os.path.join(os.path.dirname(__file__), 'test_data'),
	'noise_path': os.path.join(os.path.dirname(__file__), 'noise_dir'),
	'checkpoint_dir': 'checkpoints',
	'log_dir': 'logs',
	'epoch': 500,
	'test_steps': 64,
	'model_mode': 'train',
	'batch_size': 16,
	'seqlen': None,
	'n_vocab': None,
	'specaug_rate': 0.2,
	'freq_mask': 27,
	'time_mask': 70,
	'sample_rate': 16000, 
	'n_mels': 80,
	'n_fft': 400, # win length
	'hop_length': 160,
	'chunk_length': None,
	'n_samples': None,
	'n_frames': None,
	'accumulation_steps': 1,
	'dtype': torch.float16,
	'dim': 64,
	'nlayers': 6,
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
	'n_audio_ctx': None,

	'n_text_ctx': None,

	'causal_mode': 'non-causal', # causal, non-causal, grouped-causal, bw-semi-causal
	'variation': '',
	'model_path': '',
	'train_id': '',
	'dataset_name': 'digits',
	'nar': False, # non auto regressive
	'fine_tune': False,
	'no_footprint': False,
	'freeze_encoder': False,
	'freeze_decoder': False,
	'wandb': False,
	'save_checkpoints': True,
	'partial_test': False,
	'logger': None,
}

config = Config(params)
config.autocast = torch.autocast(device_type=config.device, dtype=config.dtype)
cache = {}


class RandomBackgroundNoise(nn.Module):
	def __init__(self,
		sample_rate: int,
		noise_dir: str,
		min_snr_db: int = 0,
		max_snr_db: int = 15,
	) -> NoReturn:
		super().__init__()

		self.sample_rate = sample_rate
		self.min_snr_db = min_snr_db
		self.max_snr_db = max_snr_db

		if not os.path.exists(noise_dir):
			raise IOError(f'Noise directory `{noise_dir}` does not exist')

		self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
		if len(self.noise_files_list) == 0:
			raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

	def __call__(self, audio_data: Tensor) -> Tensor:
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
			noise = noise[..., offset:offset + audio_length]
		elif noise_length < audio_length:
			noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim=-1)

		snr_db = random.randint(self.min_snr_db, self.max_snr_db)
		snr = math.exp(snr_db / 10)
		audio_power = audio_data.norm(p=2)
		noise_power = noise.norm(p=2)
		scale = snr * noise_power / audio_power

		return (scale * audio_data + noise ) / 2


class RandomSpeedChange(nn.Module):
	def __init__(self, sample_rate: int) -> NoReturn:
		super().__init__()
		self.sample_rate = sample_rate


	def forward(self, audio_data: Tensor) -> Tensor:
		speed_factor = random.choice([0.9, 1.0, 1.1, 1.2])
		if speed_factor == 1.0: # no change
			return audio_data

		sox_effects = [
			['speed', str(speed_factor)],
			['rate', str(self.sample_rate)],
		]
		transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
			audio_data, self.sample_rate, sox_effects)
		return transformed_audio
 

class TimeStretch(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
	
	def forward(self, x: Tensor) -> Tensor:
		factor = torch.rand(1).item() * 0.4 + 0.8
		y = torchaudio.functional.phase_vocoder(x, rate=factor, phase_advance=0.0)
		x = y.to(x.dtype)
		return x


class Augmentator(nn.Module):

	def __init__(self, rate: Optional[int] = config.specaug_rate,
			freq_mask: Optional[int] = config.freq_mask,
			time_mask: Optional[int] = config.time_mask,
		) -> NoReturn:
		super().__init__()
		self.rate = rate
		self.before = config.regularization_on_raw_audio
		self.after = config.regularization_on_mel
		# Regularization on raw audio
		self.noise = RandomBackgroundNoise(config.sample_rate, config.noise_path)
		self.speed = RandomSpeedChange(config.sample_rate)
		# Regularization on mel
		self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask)
		self.time_stretch = TimeStretch()


	def forward(self, x: Tensor, before_or_after: str) -> Tensor:
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


	def before(self, x: Tensor) -> Tensor:
		probability = torch.rand(1).item()
		if probability > self.rate:
			x = self.augment_before(x)
		return x


	def after(self, x: Tensor) -> Tensor:
		probability = torch.rand(1).item()
		if probability > self.rate:
			x = self.augment_after(x)
		return x


def load_audio(file: str, sr: int = config.sample_rate) -> np.ndarray:
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

	try:
		out = run(cmd, capture_output=True, check=True).stdout
	except CalledProcessError as e:
		raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

	return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = config.n_samples, *, axis: int = -1) -> Tensor:
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
def mel_filters(device, n_mels: int = config.n_mels) -> Tensor:
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


def apply_spec_augment(
	spectrogram,
	num_time_masks: int = 10,
	pS: int = 0.05,
	F: int = 27,
	W: int = 20,
) -> Tensor:
	'''
		Spectrogram augmentation
	'''
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
	for _ in range(np.random.randint(1, F + 1)):
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
) -> Tensor:

	audio = load_audio(audio)
	audio = torch.from_numpy(audio)

	# if augmentator is not None:
	# 	audio = augmentator(audio.view(1, -1), before_or_after='before').view(-1)

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
) -> Tensor:

	mel = log_mel_spectrogram(audio, padding=config.n_samples, device=device, augmentator=augmentator)
	mel = pad_or_trim(mel, config.n_frames)

	if augmentator:
		mel = apply_spec_augment(mel)

	return mel


def normalizer(txt: str) -> str:
	txt = re.sub(r'[\?,\.\:/\]\[\{\}\=\+\(\)\!\$\%\&\*\'\"]+', '', txt)
	return txt.lower()


def prepare_text(
	text: str,
	device: Union[str, torch.device],
) -> Tuple[Tensor, Tensor]:

	encoded = config.text_process.encoder(text, config.seqlen)
	sequence = encoded[:-1]
	labels = encoded[1:]
	sequence = torch.tensor(sequence).to(device)
	labels = torch.tensor(labels).to(device)
	return sequence, labels


class DataDigits(torch.utils.data.Dataset):
	def __init__(self, mode: Tensor, device: Union[str, torch.device]) -> NoReturn:
		self.device = device
		self.text_process = config.text_process
		self.data = []
		self.mode = mode

		self.augmentator = None
		if self.mode == 'train':
			self.augmentator = Augmentator()
			dir_path = config.train_path
		else:
			dir_path = config.test_path
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


	def __len__(self) -> int:
		return len(self.data)


	def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
		if torch.is_tensor(idx):
			idx = idx.item()

		file_path = self.data[idx]['key']
		mel_segment = prepare_audio(file_path, self.device, self.augmentator)
		text = open(self.data[idx]['text']).read()
		sequence, labels = prepare_text(text, self.device)
		return mel_segment.to(config.dtype), sequence, labels


class DataBoolq(torch.utils.data.Dataset):
	def __init__(self, mode: str, device: Union[str, torch.device]) -> NoReturn:
		self.device = device
		self.text_process = config.text_process
		self.data = []
		self.mode = mode
		ds = datasets.load_dataset('fixie-ai/boolq-audio')


		self.augmentator = None
		if self.mode == 'train':
			self.augmentator = Augmentator()
			self.data = ds['train']
		else:
			self.data = ds['validation']
		self.data = self.data.remove_columns('answer')
		self.data = self.data.remove_columns('passage')
		self.data = self.data.remove_columns('explanation')


	def __len__(self) -> int:
		return len(self.data)


	def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
		if torch.is_tensor(idx):
			idx = idx.item()

		text = normalizer(self.data[idx]['question'])
		mel_segment = prepare_audio(np.float32(self.data[idx]['audio']['array']), self.device, self.augmentator)
		sequence, labels = prepare_text(text, self.device)
		return mel_segment.to(config.dtype), sequence, labels


class DataLibSpeech100h(torch.utils.data.Dataset):

	def __init__(self, mode: str, device: Union[str, torch.device]) -> NoReturn:
		self.device = device
		self.text_process = config.text_process
		self.data = []
		self.mode = mode
		ds = datasets.load_dataset('saeedq/librispeech_100h')

		self.augmentator = None
		if self.mode == 'train':
			self.augmentator = Augmentator()
			self.data = ds['train']
		else:
			self.data = ds['test']


	def __len__(self) -> int:
		return len(self.data)


	def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
		if torch.is_tensor(idx):
			idx = idx.item()

		text = normalizer(self.data[idx]['text'])
		mel_segment = prepare_audio(np.float32(self.data[idx]['audio']['array']), self.device, self.augmentator)
		sequence, labels = prepare_text(text, self.device)
		return mel_segment.to(config.dtype), sequence, labels
