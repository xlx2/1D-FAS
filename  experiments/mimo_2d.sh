# !/bin/bash

# 激活虚拟环境（如果有）
# source ~/your_venv_path/bin/activate

# 运行 Python 脚本并传递参数
python test_mimo_2d.py \
  --num_users 5 \
  --num_y_antennas 4 \
  --num_z_antennas 4 \
  --target_pos "30,30,70;60,0,50;60,-45,30" \
  --bs_pos "0,0,0" \
  --power_dBm 30 \
  --channel_noise_dB -30 \
  --sensing_threshold_dB 0 \
  --antenna_spacing 0.5 \
  --num_trials 100
