"""Reinforcement-learning wrapper around the py-clash-bot automation stack.

The rl/ package never imports anything from the top-level pyclashbot/
namespace directly except through rl.bridge — that way upstream changes
to the scripted bot don't bleed into the RL training code.
"""
