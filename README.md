To setup the repo:
```bash

git clone https://github.com/saeeddhqan/intern-experiments; cd intern-experiments; git checkout whisper; pip install -r requirements.txt; apt install libsox-dev; wandb login;
```

To train small.en with default config: 
```bash

python main.py -a train --batch_size 7 --accumulation_steps 4 --model_path small.en --freeze_decoder --partial_test --dataset_name librispeech100h --wandb --dim 768 --nlayers 12 --epoch 20 --nheads 12
```

To check if the loaded whisper model works properly:
```bash

python main.py -a test --model_path small.en --freeze_decoder --partial_test --dataset_name librispeech100h --dim 768 --nlayers 12 --epoch 20 --nheads 12
```

To get transcripts of a local audio:
```bash

python main.py -a live --model_path \<model path\> --freeze_decoder --partial_test --dataset_name librispeech100h --dim \<model dim\> --nlayers \<model height\> --nheads \<model heads\>
```
