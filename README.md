# Satellite Finder in GALEX data

21.05.2026
@mauritz.wicker
CTAO Summer School 2026
La Palma, Canary Islands

Project: In GALEX data we want to identify variable sources, specifically differentiate Astrophysical from Non-Astrophysical variability.
Difficulties: Spiral dither pattern creates uncommon variability.
Approach:
- Bin data in RA/Dec Time
- Searching for Variability: Deviation from “Baseline” to define candidates
- Fitting Baseline and removing baseline signal.
- Periodogram analysis of remaining data to remove candidates coming from spiral dither pattern
- Identify clear >=(mu + X*sigma) outliers from binned baseline data
- Idenfity on-sky companion-bins (from Coordinates and Time evolution) [SATELLITES]
- Isolate companion-bins and extract satellite motion (RA/Dec and on-sky)
- Assuming circular orbital motion and negligagable satellite mass -> determine satellite altitude and speed (transverse)
- Make assumptions:
    - Are different trails from the same source (reflection?)
    - Is a satellite in geostationary orbit?
