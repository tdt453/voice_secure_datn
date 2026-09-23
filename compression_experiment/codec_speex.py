"""
Speex Codec
"""

import subprocess


def encode_file(input_wav, output_spx):

    subprocess.run(
        [
            "speexenc",
            "--quality",
            "8",
            input_wav,
            output_spx
        ],
        check=True
    )


def decode_file(input_spx, output_wav):

    subprocess.run(
        [
            "speexdec",
            input_spx,
            output_wav
        ],
        check=True
    )