### fish1928-2025
Welcome to Ubuntu 24.04.2 LTS (GNU/Linux 6.8.0-54-generic x86_64)
| NVIDIA-SMI 550.144.03             Driver Version: 550.144.03     CUDA Version: 12.4 |

```
install miniconda
conda activate
```
### (base)test
```
run: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
run: pip3 install ipython
run: pip3 install transformers==4.17
```

### ipython:

```
import torch
torch.cuda.is_available()
```

-----
### samples
- simple_predict_model.py -> predict
- simple_save_model.py -> load & save
- simple_train_gosv -> train gosv