from utils.math import dBm2pow, dB2pow
from Channel.fas import FAS_Channel
import numpy as np
# np.random.seed(10)  # fix the random parameters

# --------------------- System Parameters ----------------------
NUM_OF_USERS = 3  # Communication user number
NUM_OF_ANTENNAS = 16  # Antenna number
NUM_OF_SELECTED_ANTENNAS = 8  # Selected antenna number
FRAME_LENGTH = 10  # ISAC signal frame length
W = 8  # Total length of antennas (Lambda/per)

# --------------------- Power Parameters -----------------------
POWER = dBm2pow(30)  # Transmit power 
CHANNEL_NOISE = dB2pow(-30)  # Communication noise power
SENSING_NOISE = dB2pow(-30)  # Radar sensing noise power
QOS_THRESHOLD = dB2pow(12)  # QoS threshold
REFLECTION_COEFFICIENT = dB2pow(-30) # Reflection coefficient

# --------------------- Targets Parameters ---------------------
DOA = np.deg2rad([-30, 0, 30]).reshape(1, -1)  # direction of arrival
NUM_OF_TARGETS = DOA.shape[1]  # Targets number

# --------------------- Steering Vector ------------------------
antenna_indices = np.arange(-(NUM_OF_ANTENNAS - 1) / 2, (NUM_OF_ANTENNAS - 1) / 2 + 1).reshape(-1, 1)
SPACING = W / (NUM_OF_ANTENNAS-1)
STEERING_VECTOR = np.exp(1j * 2 * np.pi * antenna_indices * SPACING * np.sin(DOA))
STEERING_VECTOR_DIFF = 1j * 2 * np.pi * antenna_indices * SPACING * np.cos(DOA) * STEERING_VECTOR

# --------------------- Channel --------------------------------
fas = FAS_Channel(Ny=NUM_OF_ANTENNAS, N_user=NUM_OF_USERS, Wx=W, Wy=W)
CHANNEL= fas.get_channel()

# --------------------- Parameters ----------------------
Parameters = {
    "K": NUM_OF_USERS,
    "M": NUM_OF_TARGETS,
    "N": NUM_OF_ANTENNAS,
    "Nt": NUM_OF_SELECTED_ANTENNAS,
    "spacing": SPACING,
    "L": FRAME_LENGTH,
    "P": POWER,
    "Gamma": QOS_THRESHOLD,
    "sigmaC2": CHANNEL_NOISE,
    "sigmaR2": SENSING_NOISE,
    "rc": REFLECTION_COEFFICIENT,
    "a": STEERING_VECTOR,
    "a_diff": STEERING_VECTOR_DIFF,
    "H": CHANNEL,
}



