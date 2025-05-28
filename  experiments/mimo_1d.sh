# !/bin/bash

# 激活虚拟环境（如果有）
# source ~/your_venv_path/bin/activate

# 运行 Python 脚本并传递参数
python test_mimo_1d.py \
  --num_users 5 \
  --num_antennas 8 \
  --frame_length 10 \
  --power_dBm 30 \
  --channel_noise_dB -30 \
  --sensing_noise_dB -30 \
  --qos_threshold_dB 8 \
  --reflection_coeff_dB -30 \
  --doa_degrees -30 0 30 \
  --antenna_spacing 0.5 \
  --num_trials 100
