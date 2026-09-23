from compression_experiment.crypto_codec_pipeline import run_opus_crypto

stats = run_opus_crypto(
    "data/voice.wav",
    "codec_results/opus"
)

print(stats)