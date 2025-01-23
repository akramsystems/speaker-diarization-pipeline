import os
import matplotlib.pyplot as plt
import torch
from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate, GreedyMapper
from pyannote.core import Annotation, Segment
from pydub import AudioSegment

from src.dataloader import DataLoader
from src.config import config


class SpeakerDiarizationPipeline:
    def __init__(self, audio_file, true_rttm_path, output_dir):
        # Convert mp3 to wav
        if audio_file.endswith('.mp3'):
            audio_file_wav = audio_file.replace('.mp3', '.wav')
            if not os.path.exists(audio_file_wav):
                audio = AudioSegment.from_mp3(audio_file)
                audio.export(audio_file_wav, format='wav')
            audio_file = audio_file_wav
        
        self.audio_file = audio_file
        self.audio_filename = os.path.splitext(os.path.basename(self.audio_file))[0]
        self.true_rttm_path = true_rttm_path
        self.output_dir = output_dir
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=config.hf_token)
        self.num_speakers = self.get_number_of_speakers()
    
    def get_number_of_speakers(self) -> int:
        speakers = set()
        with open(self.true_rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                speaker = parts[7]
                speakers.add(speaker)
        return len(speakers)

    def diarize(self) -> Annotation:
        self.pipeline.to(torch.device("mps"))
        diarization = self.pipeline(self.audio_file, num_speakers=self.num_speakers)
        return diarization

    def save_predicted_rttm(self, diarization: Annotation):
        pred_rttm_dir = os.path.join(self.output_dir, "pred_rttm")
        os.makedirs(pred_rttm_dir, exist_ok=True)
        
        # Use the audio file name for the predicted RTTM file
        pred_rttm_path = os.path.join(pred_rttm_dir, f"{self.audio_filename}.rttm")
        
        with open(pred_rttm_path, 'w') as f:
            diarization.write_rttm(f)
        print(f"Predicted RTTM saved to {pred_rttm_path}")

    def calculate_der(self, diarization: Annotation):
        # Load ground truth RTTM
        reference = Annotation()
        with open(self.true_rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
                reference[Segment(start, start + duration)] = speaker

        # Map predicted speakers to reference speakers
        # mapping = GreedyMapper()
        # mapped_diarization = mapping(reference, diarization)
        # Ensure mapped_diarization is an Annotation object
        # if isinstance(mapped_diarization, dict):
        #     mapped_diarization = Annotation.from_dict(mapped_diarization)

        # Calculate DER
        metric = DiarizationErrorRate()
        # Has a Default Mapper
        der = metric(reference, diarization)
        print(f"Diarization Error Rate (DER): {der * 100:.2f}%")

        # Calculate percentage of audio diarized
        total_duration = sum(segment.duration for segment in reference.get_timeline())
        diarized_duration = sum(segment.duration for segment in diarization.get_timeline())
        percentage_diarized = diarized_duration / total_duration

        # Plot DER vs. Percentage of Audio Diarized
        plt.figure()
        plt.plot([percentage_diarized], [der * 100], 'bo')
        plt.xlabel('Percentage of Audio Diarized')
        plt.ylabel('Diarization Error Rate (%)')
        plt.title('DER vs. Percentage of Audio Diarized')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        plt.grid(True)
        plt.show()

        return der

if __name__ == "__main__": 
    # Paths
    audio_file = "dataset/audio/11.mp3"
    transcript_file = "dataset/transcript/11.json"
    output_dir = "outputs"

    # Initialize DataLoader and generate RTTM
    data_loader = DataLoader(audio_file, transcript_file, output_dir)
    data_loader.generate_true_rttm()
    true_rttm_path = data_loader.get_true_rttm_path()

    # Initialize and execute the pipeline
    pipeline = SpeakerDiarizationPipeline(audio_file, true_rttm_path, output_dir)
    diarization = pipeline.diarize()
    pipeline.save_predicted_rttm(diarization)
    pipeline.calculate_der(diarization)
