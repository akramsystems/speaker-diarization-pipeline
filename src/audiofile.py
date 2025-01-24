import os
import json
from pydub import AudioSegment
from pyannote.core import Annotation, Segment


class AudioFile:
    def __init__(self, audio_file_path, transcript_file_path, output_dir):
        self.transcript_file_path = transcript_file_path
        self.audio_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        
        self.output_dir = os.path.abspath(output_dir)
        self.true_rttm_path = os.path.join(self.output_dir, "true_rttm", f"{self.audio_filename}.rttm")
        self.pred_rttm_path = os.path.join(self.output_dir, "pred_rttm", f"{self.audio_filename}.rttm")
        self.diarization_path = os.path.join(self.output_dir, "diarization_results", f"{self.audio_filename}.pkl")
        self.accuracy_plot_path = os.path.join(self.output_dir, "accuracy_plots", f"{self.audio_filename}_der_plot.png")
        
        os.makedirs(os.path.dirname(self.true_rttm_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.pred_rttm_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.diarization_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.accuracy_plot_path), exist_ok=True)

        self.generate_true_rttm()
        self.num_speakers = self.get_number_of_speakers()
        self.audio_file_path = self.convert_audio_to_wav(audio_file_path)
        self.reference = self.get_reference_annotation()
    
    def get_reference_annotation(self):
        # Load ground truth RTTM
        reference = Annotation()
        with open(self.true_rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
                reference[Segment(start, start + duration)] = speaker
        return reference

    def convert_audio_to_wav(self, audio_file: str) -> str:
        if audio_file.endswith('.mp3'):
            audio_file_wav = audio_file.replace('.mp3', '.wav')
            if not os.path.exists(audio_file_wav):
                audio = AudioSegment.from_mp3(audio_file)
                audio.export(audio_file_wav, format='wav')
            return audio_file_wav
        return audio_file

    def get_number_of_speakers(self) -> int:
        speakers = set()
        with open(self.true_rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                speaker = parts[7]
                speakers.add(speaker)
        return len(speakers)

    def generate_true_rttm(self):
        if not os.path.exists(self.true_rttm_path):
            with open(self.transcript_file_path, 'r') as f:
                transcript_data = json.load(f)
            
            with open(self.true_rttm_path, 'w') as rttm_file:
                for _, entries in transcript_data.items():
                    for entry in entries:
                        speaker_name = entry['speaker'].replace(' ', '_')
                        rttm_file.write(
                            f"SPEAKER ep-{entry['episode']} 1 {entry['utterance_start']} "
                            f"{entry['duration']} <NA> <NA> {speaker_name} <NA> <NA>\n"
                        )
            print(f"True RTTM file created at {self.true_rttm_path}")
        else:
            print(f"True RTTM file already exists at {self.true_rttm_path}")

    def get_true_rttm_path(self):
        return self.true_rttm_path

if __name__ == "__main__":
    audio_file_path = "dataset/audio/11.mp3"
    transcript_file_path = "dataset/transcript/11.json"
    output_dir = "outputs"
    audio_file_instance = AudioFile(audio_file_path, transcript_file_path, output_dir)
    audio_file_instance.generate_true_rttm()
    true_rttm_path = audio_file_instance.get_true_rttm_path()
    print(f"True RTTM file path: {true_rttm_path}") 