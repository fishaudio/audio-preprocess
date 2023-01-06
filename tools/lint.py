import subprocess as sp

# Black
sp.run(["black", "fish_audio_preprocess", "tools"])

# Isort
sp.run(["isort", "fish_audio_preprocess", "tools"])
