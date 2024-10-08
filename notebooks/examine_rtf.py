import whisper
import time, math


model = whisper.load_model('small.en')
options = whisper.DecodingOptions(language='en')
is_stream = True
audio = whisper.load_audio('Once.m4a')

'''
value for samples:
	benchmarking one second streaming: {100: (1, 30, 50)}
	benchmarking two seconds streaming: {200: (1, 15, 100)}
	benchmarking two and half seconds streaming: {250: (1, 12, 125)}
	benchmarking three seconds streaming: {300: (1, 10, 150)}
	benchmakring six seconds streaming: {600: (1, 5, 300)}
'''
samples = {100: (1, 30, 50)}
results = {x: -1 for x in samples}
steps = 50
for sample in samples:
	pad, n_steps, n_samples = samples[sample]
	took = 0.0
	model.encoder.reset_states()
	if is_stream:
		audio_tmp = whisper.pad_or_trim(audio)
		audio_tmp = whisper.log_mel_spectrogram(audio_tmp).to(model.device)
		model.encoder.max_step = n_steps
		chunks = []
		for chunk in range(n_steps - 1):
			t = audio_tmp[:,chunk * sample: (chunk * sample) + sample + pad].unsqueeze(0)
			t = model.encoder(t, streaming=True, step=chunk, samples=n_samples)
			chunks.append(t)
		chunk = n_steps - 1
		t = audio_tmp[:,chunk * sample: (chunk * sample) + sample + pad].unsqueeze(0)
		for _ in range(steps):
			tmp_chunks = chunks.copy()
			start = time.perf_counter()
			tmp_chunks.append(model.encoder(t, streaming=True, step=-1, samples=n_samples))
			audio_tmp = whisper.torch.cat(tmp_chunks, dim=1)
			end = time.perf_counter()
			took += end - start
	else:
		audio_tmp = whisper.pad_or_trim(audio)
		audio_tmp = whisper.log_mel_spectrogram(audio_tmp).to(model.device).unsqueeze(0)
		for _ in range(steps):
			audio_tmp_2 = audio_tmp
			start = time.perf_counter()
			audio_tmp_2 = model.encoder(audio_tmp_2)
			end = time.perf_counter()
			took += end - start
		audio_tmp = audio_tmp_2
	took /= steps
	result = whisper.decode(model, audio_tmp, options)
	print('stream:', n_pads)
	print('\tRTF:', took / 30)
	print('\ttook:', took)
	print('\tframe size:', sample / 100, 's')
	print(result[0].text)
