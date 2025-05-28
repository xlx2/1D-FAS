# !/bin/bash

# 激活虚拟环境（如果有）
# source ~/your_venv_path/bin/activate

# 运行 Python 脚本并传递参数
python test_fas_1d.py \
  --num_users 5 \
  --num_antennas 16 \
  --num_selected_antennas 8 \
  --frame_length 10 \
  --W 7.5 \
  --power_dBm 30 \
  --channel_noise_dB -30 \
  --sensing_noise_dB -30 \
  --qos_threshold_dB 16 \
  --reflection_coeff_dB -30 \
  --doa_degrees -30 0 30 \
  --num_trials 100