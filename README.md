# Beat-tracking-system

This repository contains the implementation of a beat tracking system, developed as part of the Music Informatics module for the MSc Sound and Music Computing programme at Queen Mary University of London. 

The system in the present work is aimed specifically for ballroom dance music, and has three main parts: onset detection, tempo induction, and beat tracking.

The two scripts included are:
- beat_detection_for_running.py: script for running all components of the system to obtain the estimated beat times.
- beat detection evaluation.ipynb: script that was used during developing, containing the steps in beat_detection_for_running.py, as well as the evaluation code.

The data used for testing are the audio clips available at http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html.

Ground truth annotations of beats for this dataset are provided by Florian Krebs here: https://github.com/CPJKU/BallroomAnnotations

More details on the data and the mechanics of the system are given in the written report available in this repository.
