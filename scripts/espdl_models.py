# PlatformIO extra script: generate the embedded face-detection model.
#
# The human_face_detect component packs its .espdl model files and turns them
# into an assembly file (human_face_detect.espdl.S) with CMake custom commands.
# PlatformIO compiles the sources CMake reports but never runs those commands,
# so this script does the same two steps before the build starts.

import os
import subprocess
import sys

Import("env")

PROJECT_DIR = env.subst("$PROJECT_DIR")
BUILD_DIR = env.subst("$BUILD_DIR")
COMPONENTS = os.path.join(PROJECT_DIR, "managed_components")
ESPDL_DIR = os.path.join(COMPONENTS, "espressif__esp-dl")
MODELS_DIR = os.path.join(COMPONENTS, "espressif__human_face_detect", "models", "s3")

PACKED = os.path.join(BUILD_DIR, "espdl_models", "human_face_detect.espdl")
ASM = os.path.join(BUILD_DIR, "human_face_detect.espdl.S")

# Which models the component's Kconfig says to flash -> files to pack
MODEL_FILES = {
    "CONFIG_FLASH_HUMAN_FACE_DETECT_MSRMNP_S8_V1": [
        "human_face_detect_msr_s8_v1.espdl",
        "human_face_detect_mnp_s8_v1.espdl",
    ],
    "CONFIG_FLASH_ESPDET_PICO_224_224_FACE": ["espdet_pico_224_224_face.espdl"],
    "CONFIG_FLASH_ESPDET_PICO_416_416_FACE": ["espdet_pico_416_416_face.espdl"],
}


def enabled_models():
    sdkconfig = os.path.join(PROJECT_DIR, "sdkconfig." + env.subst("$PIOENV"))
    with open(sdkconfig) as f:
        enabled = {line.split("=")[0] for line in f if line.strip().endswith("=y")}
    files = []
    for option, names in MODEL_FILES.items():
        if option in enabled:
            files += [os.path.join(MODELS_DIR, n) for n in names]
    return files


def is_stale(target, sources):
    if not os.path.isfile(target):
        return True
    built = os.path.getmtime(target)
    return any(os.path.getmtime(s) > built for s in sources)


def generate():
    if not os.path.isdir(MODELS_DIR):
        return  # component not downloaded yet; CMake will fetch it on this run

    models = enabled_models()
    if not models:
        sys.stderr.write("Warning! No human_face_detect model enabled in sdkconfig\n")
        return

    os.makedirs(os.path.dirname(PACKED), exist_ok=True)
    if is_stale(PACKED, models):
        print("Packing face detection model...")
        subprocess.check_call([
            sys.executable,
            os.path.join(ESPDL_DIR, "fbs_loader", "pack_espdl_models.py"),
            "--model_path", *models,
            "--out_file", PACKED,
        ])

    embed_script = os.path.join(ESPDL_DIR, "fbs_loader", "cmake", "data_file_embed_asm_aligned.cmake")
    if is_stale(ASM, [PACKED, embed_script]):
        print("Generating assembly for face detection model...")
        cmake = os.path.join(env.PioPlatform().get_package_dir("tool-cmake"), "bin", "cmake")
        subprocess.check_call([
            cmake,
            "-DDATA_FILE=" + PACKED.replace("\\", "/"),
            "-DSOURCE_FILE=" + ASM.replace("\\", "/"),
            "-DFILE_TYPE=BINARY",
            "-P", embed_script,
        ], cwd=BUILD_DIR)


generate()
