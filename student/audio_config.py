"""Single source of truth for the 8 kHz framing.

The z latent is fixed at the teacher's frame rate (44100/512). 8000 is NOT an integer
multiple of it, so we resample z in the latent domain to an intermediate frame rate that
divides 8000 cleanly, then upsample by an integer factor (iSTFT hop) to land on exactly
8000 Hz. These constants are imported by the data dump, the trainers, and the ONNX export
so every stage uses identical framing.
"""
TEACHER_SR = 44100
HOP_TEACHER = 512
Z_FRAME_RATE = TEACHER_SR / HOP_TEACHER       # 86.1328125 Hz  (fixed by enc/flow)
Z_CHANNELS = 192
G_CHANNELS = 256

TARGET_SR = 8000
INTER_FRAME_RATE = 125.0                        # 8000 / 64; divides TARGET_SR cleanly
NFFT = 256                                      # Nyquist 4000 Hz covers <=3400 telephony band
HOP = 64                                        # TARGET_SR / INTER_FRAME_RATE -> exact 8000
assert TARGET_SR % HOP == 0 and TARGET_SR // HOP == int(INTER_FRAME_RATE)

# z -> intermediate frame-rate resample factor. Used as F.interpolate(scale_factor=...) so the
# exported decoder graph needs ONLY (z, g) -- the output length is floor(T*RESAMPLE_SCALE),
# matching ONNX Resize-with-scales exactly (same rule in training and inference).
RESAMPLE_SCALE = INTER_FRAME_RATE / Z_FRAME_RATE   # 1.45124716...


def inter_frames_for(z_frames: int) -> int:
    """INTER_FRAME_RATE frames produced from z_frames z-frames (floor, matches Resize-scales)."""
    return int(z_frames * RESAMPLE_SCALE)


def out_samples_for(z_frames: int) -> int:
    """Final 8 kHz sample count = inter_frames * HOP (iSTFT tail trimmed)."""
    return inter_frames_for(z_frames) * HOP
