"""Φ — Physical Hardware Intelligence.

A thin, well-documented layer over Hugging Face LeRobot for the Northeastern SV
robotics SIG. Subpackages:

    data    — dataset conversion, validation, stats, augmentation
    train   — wrappers around `lerobot-train` per policy
    eval    — standardized rollout evaluation + metrics + reports
    deploy  — on-robot / remote / edge inference
    utils   — logging, seeding, checkpoint-resume, wandb helpers

Design rule: prefer calling upstream LeRobot over copying it. Keep this thin.
"""

__version__ = "0.0.1"
