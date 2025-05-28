import argparse
import numpy as np
from utils.math import dBm2pow, dB2pow
from LA_FAS.system import FA_1D_System
from Channel.fas import FAS_Channel
np.random.seed(10)

def get_args():
    parser = argparse.ArgumentParser(description="Fluid Antenna ISAC system configuration")
    parser.add_argument("--num_users", type=int, default=3, help="Number of communication users (K)")
    parser.add_argument("--num_antennas", type=int, default=16, help="Number of transmit antennas (N)")
    parser.add_argument("--num_selected_antennas", type=int, default=8, help="Number of selected transmit antennas")
    parser.add_argument("--frame_length", type=int, default=10, help="ISAC signal frame length (L)")
    parser.add_argument("--W", type=float, default=7.5, help="Total antenna length per lambda")
    parser.add_argument("--power_dBm", type=float, default=30, help="Transmit power in dBm")
    parser.add_argument("--channel_noise_dB", type=float, default=-30, help="Channel noise power in dB")
    parser.add_argument("--sensing_noise_dB", type=float, default=-30, help="Sensing noise power in dB")
    parser.add_argument("--qos_threshold_dB", type=float, default=12, help="QoS threshold in dB")
    parser.add_argument("--reflection_coeff_dB", type=float, default=-30, help="Reflection coefficient in dB")
    parser.add_argument("--doa_degrees", type=float, nargs='+', default=[-30, 0, 30], help="List of DOAs in degrees")
    parser.add_argument("--num_trials", type=int, default=200, help="Number of trials")

    return parser.parse_args()

def get_parameters(args):
    doa = np.deg2rad(args.doa_degrees).reshape(1, -1)
    antenna_indices = np.arange(-(args.num_antennas - 1) / 2, (args.num_antennas - 1) / 2 + 1).reshape(-1, 1)
    spacing = args.W / (args.num_antennas - 1)

    a = np.exp(1j * 2 * np.pi * antenna_indices * spacing * np.sin(doa))
    a_diff = 1j * 2 * np.pi * antenna_indices * spacing * np.cos(doa) * a

    H = FAS_Channel(Ny=args.num_antennas, N_user=args.num_users, Wx=args.W, Wy=args.W).get_channel()

    Parameters = {
        "K": args.num_users,
        "M": doa.shape[1],
        "N": args.num_antennas,
        "Nt": args.num_selected_antennas,
        "L": args.frame_length,
        "spacing": spacing,
        "P": dBm2pow(args.power_dBm),
        "Gamma": dB2pow(args.qos_threshold_dB),
        "sigmaC2": dB2pow(args.channel_noise_dB),
        "sigmaR2": dB2pow(args.sensing_noise_dB),
        "rc": dB2pow(args.reflection_coeff_dB),
        "theta": doa,
        "a": a,
        "a_diff": a_diff,
        "H": H
    }

    return Parameters

def run_montecarlo_mimo(Parameters, n_trials):
    mimo_parameters = Parameters.copy()
    crb_results = []

    for i in range(n_trials):
        try:
            print(f"[{i+1}/{n_trials}]")
            mimo_parameters['H'] = FAS_Channel(Ny=args.num_antennas, N_user=args.num_users, Wx=args.W, Wy=args.W).get_channel()
            system = FA_1D_System(**mimo_parameters)
            _, crb = system.run()
            crb_results.append(crb)
        except Exception as e:
            print(f"[Warning] Iteration {i+1} failed: {e}")
            continue

    print(f"\nSuccess: {len(crb_results)} / {n_trials}")
    if crb_results:
        mean_crb = sum(crb_results) / len(crb_results)
        print(f"Mean CRB: {mean_crb:.4f}")

if __name__ == '__main__':
    args = get_args()
    Parameters = get_parameters(args)
    run_montecarlo_mimo(Parameters, args.num_trials)
