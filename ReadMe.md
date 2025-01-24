# Speaker Diarization

- Python3.11
## Overview

This project implements a speaker diarization system using the `pyannote.audio` library. The system processes audio files to identify and differentiate between speakers, providing a diarization error rate (DER) and jaccard error rate (JER) plots to evaluate performance. We use the pipeline from huggingface to perform the speaker diarization, since it combines both the segmentation and speaker diarization in one pipeline for us.  A more detailed explanation of the pipeline can be found [here](https://huggingface.co/pyannote/speaker-diarization-3.1).

The TLDR of the speaker diarization pipeline can be observed in the following image:

![Speaker Diarization](imgs/speaker-diarization-process.svg)

## Features

- **Speaker Diarization**: Identifies and segments speakers in an audio file.
- **Diarization Error Rate (DER) Calculation**: Computes the DER to evaluate the accuracy of the diarization.
- **Streamlit Integration**: Provides a web interface for uploading audio and transcript files and visualizing results.
- **Device Compatibility**: Automatically selects the best available device (CPU, CUDA, or MPS) for processing. (note: CUDA is recommended for faster processing)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/akramsystems/speaker-diarization-pipeline.git
   cd speaker-diarization-pipeline
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add your Hugging Face token:
     ```
     HF_TOKEN=your_huggingface_token
     ```
    make sure to accept the terms of using the speaker diarization model from Hugging Face [link here](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Usage

### Streamlit Web Interface

1. **Start the Streamlit app**:
   ```bash
   python -m streamlit run src/app.py
   ```

2. **Upload files**:
   - Use the web interface to upload an audio file and its corresponding transcript, the pipeline will automatically process once both files are uploaded.

    ![Streamlit Upload Files](imgs/streamlit-upload-files.png)

3. **Wait for the pipeline to finish**:
   - It takes a while, I was using an m1 2020 macbook pro, it took about 10 minutes.

    ![Streamlit Progress Bar](imgs/glitchy-loading-bar.png)

3. **View results**:
   - The app will display a plot of the Diarization Error Rate (DER) over the percentage of audio processed.
   - it should also save the plot to the `outputs/accuracy_plots` directory.

   (pretty underwhelming results I agree)

    ![Streamlit DER Plot](imgs/streamlit-der-plot.png)
    ![Streamlit JER Plot](imgs/streamlit-jer-plot.png)

    ```
    Note: If you repeat the experiment with the same files it will load the same previous diariazation results since i pickle it and save it in the /outputs/diarization_results folder since my computer was slow, and debugging was slow. If you want fresh results delete the saved diarizations and the pipeline will re-run for your given audio file.
    ```

## Project Structure

- `src/`: Contains the main source code.
  - `audiofile.py`: Handles audio file processing.
  - `pipeline.py`: Implements the speaker diarization pipeline.
  - `config.py`: Manages configuration settings.
  - `util.py`: Contains utility functions, including Streamlit progress hooks.
- `dataset/`: Directory for uploading/storing audio and transcript files.
- `outputs/`: Directory for storing output files, including RTTM file and DER + JER plots.
- `.streamlit/config.toml`: Configuration for Streamlit.


## Considerations For Future Work

The results are pretty underwhelming, but this is to be expected!  If we look at the % DER in the original [repo](https://github.com/pyannote/pyannote-audio) we see the accuracies vary between __%8 to as high as %50__ with a majority of results falling in the bounds of our accuracy (%20 - %30).  I was originally going to use the `nemo-toolkit[asr]` module since it offers a similar embedding model (TitaNet) to that of the pyannote but the clustering technique and voice activation detection (VAD) models are different.  Comparing these two approaches i can see what pipeline is better out of the box. After that Hyperparameter tuning can be done on the VAD/Segmentation proces (i.e. adjust thresholds for detection), the embedding model (use different segment lengths), and the clustering model can be tuned or replaced with a different clustering model.  I did virtually no alteration to the defaults.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
