#!/bin/bash

#python test.py  --config-name=conf_piano.yaml tester=piano_evaluator_BABE2 tester.checkpoint="experiments/MAESTRO_22kHz_8s-850kits.pt" id="BABE2" tester.evaluation.single_recording="test_examples/piano/Rachmaninoff_denoised_cropped.wav"

python test.py  --config-name=conf_singing_voice.yaml tester=singer_evaluator_BABE2 tester.checkpoint="experiments/VocalSet_male2_44kHz_6s-8kits.pt" id="BABE2" tester.evaluation.single_recording="test_examples/caruso/denoised_vocals_cropped.wav"
