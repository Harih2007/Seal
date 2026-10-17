# Seal — underwater fish detection and tracking

A local web prototype for fish detection, ByteTrack tracking, optional closed-set species classification, video annotation, and CHE01-oriented statistics and exports. Detection and tracking positions are image-pixel estimates; they are not calibrated distances. Classification results are model candidates and require expert validation.

## Run locally

1. Use Python 3.10+ and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Obtain the Fishial detector checkpoint/archive from the [Fishial model repository](https://github.com/Wye-Foundation/Fishial-Fish-Identification) and place the compatible archive at `weights/detector_v26_n3.zip`. Model weights are intentionally not committed here. You can keep weights elsewhere and set `SEAL_DETECTOR_WEIGHTS` to the detector path.
3. Start the local browser app:

   ```powershell
   python web_server.py
   ```

4. Open <http://127.0.0.1:8502/>. Set `SEAL_PORT` to use a different local port.

Results are stored in `exports/web_runs/`. To enable the optional Fishial species classifier, place its compatible TorchScript model at `classifier_model/model.pt` or set `SEAL_SPECIES_MODEL` to its path. The species classifier is not a fish detector and is not included in this repository.

The Species page can compare candidate labels with OBIS occurrence records after you enter camera longitude and latitude. The report includes that location context only after a successful lookup. Occurrences are reported observations, not confirmed range boundaries, migration routes, or proof of native/invasive status. See the [OBIS data access guide](https://manual.obis.org/access).

## Legacy Streamlit prototype

`app.py` contains the earlier local Streamlit interface. Start it with `streamlit run app.py`; upload a compatible detector checkpoint and video through that interface.

## Validation and claims

Detection metrics (precision, recall, mAP@50, mAP@50–95) require labelled validation boxes for the evaluated video domain. HOTA, IDF1, and MOTA require ground-truth tracks. The app does not treat runtime statistics or unlabelled-video predictions as accuracy metrics.
