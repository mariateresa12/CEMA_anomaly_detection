import os
import sys
import json
import pyshark
import pandas as pd

import attacks_references


def is_true(v: object) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def is_false(v: object) -> bool:
    return str(v).strip().lower() in ("0", "false", "no")


# Duraciones (segundos):
# - A y B ~30s
# - C, D, E ~60s
ATTACK_DURATION_S: dict[str, int] = {
    "A": 30,
    "B": 30,
    "C": 60,
    "D": 60,
    "E": 60,
}

DEV_ID: dict[int, str] = {
    0x0A12: "MS1",
    0x0C06: "SP2",
    0x22FD: "IB1",
    0x46EA: "DS1",
    0x5EBA: "CB2",
    0x7B9E: "WB2",
    0x7C77: "WB1",
    0x88C2: "DS2",
    0xA2AB: "SP1",
    0xA8DF: "CB1",
}

# Además de caer en la ventana temporal,
# solo se etiqueta el frame si coincide con el origen/destino esperado.
#
# - A y B: no hay una pareja SRC16/DST16 fija (el atacante usa direcciones aleatorias),
#          así que se mantienen solo por ventana temporal.
# - C: replay haciéndose pasar por IB1 hacia CB2. Además, en la captura replayada
#      aparecen Data Request de MS1 hacia WB1 como efecto lateral.
# - D: flooding haciéndose pasar por DS1 hacia CB1.
# - E: flooding haciéndose pasar por MS1 hacia WB1.
ATTACK_DEVICE_RULES: dict[str, list[tuple[str | None, str | None]]] = {
    "A": [(None, None)],
    "B": [(None, None)],
    "C": [("IB1", "CB2"), ("MS1", "WB1")],
    "D": [("DS1", "CB1")],
    "E": [("MS1", "WB1")],
}


def ts_to_epoch_us(ts: pd.Timestamp) -> int:
    """
    Convierte pandas.Timestamp a epoch en microsegundos (UTC).
    """
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize("UTC")
    else:
        ts_utc = ts.tz_convert("UTC")

    return int(ts_utc.value // 1000)


def build_attack_windows() -> list[tuple[str, int, int]]:
    """
    Devuelve lista de (tipo, start_us, end_us) para cada ataque.
    """
    windows: list[tuple[str, int, int]] = []
    for a in attacks_references.all_attacks:
        t = a["type"]
        if t not in ATTACK_DURATION_S:
            continue
        start_us = ts_to_epoch_us(a["start"])
        end_us = start_us + ATTACK_DURATION_S[t] * 1_000_000
        windows.append((t, start_us, end_us))
    return windows


def matches_device_rule(attack_type: str, src_dev: str, dst_dev: str) -> bool:
    """
    Comprueba si el par origen-destino del frame encaja con el ataque.
    """
    for expected_src, expected_dst in ATTACK_DEVICE_RULES.get(attack_type, []):
        src_ok = expected_src is None or src_dev == expected_src
        dst_ok = expected_dst is None or dst_dev == expected_dst
        if src_ok and dst_ok:
            return True
    return False


def compute_attack(epoch_us: int, src16: int, dst16: int, windows: list[tuple[str, int, int]]) -> str:
    """
    Etiqueta un frame como ataque solo si:
      1) cae dentro de la ventana temporal del ataque, y
      2) coincide con el origen/destino esperado para ese ataque.
    """
    src_dev = DEV_ID.get(src16, "UNKNOWN")
    dst_dev = DEV_ID.get(dst16, "UNKNOWN")

    for t, start_us, end_us in windows:
        if start_us <= epoch_us < end_us and matches_device_rule(t, src_dev, dst_dev):
            return t

    return "NO_ATTACK"


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: {} <file.pcap> <rpi>".format(sys.argv[0]))
        exit(1)

    c = pyshark.FileCapture(
        input_file=sys.argv[1],
        tshark_path=r"C:\Program Files\Wireshark\tshark.exe",
        custom_parameters=['-o', 'wpan.fcs_format:cc24xx']
    )
    rpi = sys.argv[2]

    rpi_values = ['1', '2', '3', '4']
    if rpi not in rpi_values:
        print("Invalid RPI value. Expected one of: {}".format(rpi_values))
        exit(1)

    attack_windows = build_attack_windows()

    print()
    print('-------------------------------------------------------------------------------------------')
    print('A RSSI extractor on IEEE 802.15.4 pcap files featuring "TI CC24xx metadata" as "FCS Format"')
    print('-------------------------------------------------------------------------------------------')
    print()

    attack_packet_count = 0
    wpan_no_fcs_count = 0
    wpan_fcs_ok_count = 0
    wpan_bad_fcs_count = 0
    wpan_src16_count = 0
    wpan_no_src16_count = 0
    wpan_security_count = 0
    wpan_bad_frame_type_count = 0
    device_addresses = set()
    retained_frames = []

    for pkt_index, pkt in enumerate(c):
        if hasattr(pkt.wpan, "fcs_ok"):
            if is_true(pkt.wpan.fcs_ok):
                wpan_fcs_ok_count += 1

                if hasattr(pkt.wpan, "src16"):
                    wpan_src16_count += 1
                    address = pkt.wpan.src16

                    if is_false(pkt.wpan.security):
                        frame_type = int(pkt.wpan.frame_type, base=16)
                        if 0 <= frame_type <= 3:
                            device_addresses.add(address)

                            src16_int = int(pkt.wpan.src16, 0)
                            dst16_int = int(pkt.wpan.dst16, 0) if hasattr(pkt.wpan, "dst16") else 0
                            length = int(pkt.frame_info.len)
                            time_relative_us = int(1e6 * float(pkt.frame_info.time_relative))
                            rssi_dbm = int(pkt.wpan.rssi) - 73
                            epoch_us = int(1e6 * float(pkt.frame_info.time_epoch))

                            attack = compute_attack(epoch_us, src16_int, dst16_int, attack_windows)
                            dev = DEV_ID.get(src16_int, "UNKNOWN")

                            retained_frames.append(
                                [
                                    pkt_index,
                                    src16_int,
                                    dst16_int,
                                    length,
                                    time_relative_us,
                                    rssi_dbm,
                                    epoch_us,
                                    attack,
                                    dev,
                                    rpi,
                                ]
                            )

                            if attack != "NO_ATTACK":
                                attack_packet_count += 1
                        else:
                            print('Packet {:7d}: has type btw 4 and 7.'.format(pkt_index))
                            wpan_bad_frame_type_count += 1
                    else:
                        print('Packet {:7d}: has wpan.security = 1.'.format(pkt_index))
                        wpan_security_count += 1
                else:
                    wpan_no_src16_count += 1
            else:
                print('Packet {:7d}: bad FCS occured.'.format(pkt_index))
                wpan_bad_fcs_count += 1
        else:
            print('Packet {:7d}: has no wpan.fcs_ok field, probably a malformed packet.'.format(pkt_index))
            wpan_no_fcs_count += 1

    wpan_packet_count = pkt_index + 1
    print()
    print('Capture {}: {} frames'.format(c, wpan_packet_count))
    print()
    print('Processed {} frames:'.format(wpan_packet_count))
    print('{:7d}\t{:8.2%}\tNo FCS (malformed packet)'.format(wpan_no_fcs_count, wpan_no_fcs_count / wpan_packet_count))
    print('{:7d}\t{:8.2%}\tBAD FCS'.format(wpan_bad_fcs_count, wpan_bad_fcs_count / wpan_packet_count))
    print('{:7d}\t{:8.2%}\tSRC16 address'.format(wpan_src16_count, wpan_src16_count / wpan_packet_count))
    print('\t(among them: {:7d} Security Enabled frames, not retained)'.format(wpan_security_count))
    print('\t(among them: {:7d} Bad Frame Type frames, not retained)'.format(wpan_bad_frame_type_count))
    print('{:7d}\t{:8.2%}\tNo SRC16 address'.format(wpan_no_src16_count, wpan_no_src16_count / wpan_packet_count))
    wpan_packet_count_check = wpan_no_fcs_count + wpan_bad_fcs_count + wpan_src16_count + wpan_no_src16_count
    print('{:7d}\t{:8.2%}\tTOTAL'.format(wpan_packet_count_check, wpan_packet_count_check / wpan_packet_count))
    print()
    print('Found following SRC16 addresses:')
    print(device_addresses)
    print()
    print('Found {} frames with attacks.'.format(attack_packet_count))
    print()
    print('Working on {} light frames.'.format(len(retained_frames)))
    print(retained_frames[:3])
    print('...')
    print()
    temp = os.path.splitext(sys.argv[1])[0]
    base = os.path.basename(temp)
    filename = base + '.json'
    with open(filename, 'w') as f:
        json.dump(retained_frames, f)
    print("Created file {}".format(filename))

    c.close()


if __name__ == "__main__":
    main()
