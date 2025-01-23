import os
import json

class DataLoader:
    def __init__(self, audio_file, transcript_file, output_dir):
        self.audio_file = audio_file
        self.transcript_file = transcript_file
        self.output_dir = output_dir

        audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
        self.true_rttm_path = os.path.join(output_dir, "true_rttm", f"{audio_filename}.rttm")
        
        os.makedirs(os.path.dirname(self.true_rttm_path), exist_ok=True)

    def generate_true_rttm(self):
        if not os.path.exists(self.true_rttm_path):
            with open(self.transcript_file, 'r') as f:
                transcript_data = json.load(f)
            
            with open(self.true_rttm_path, 'w') as rttm_file:
                for _, entries in transcript_data.items():
                    for entry in entries:
                        # Replace spaces in speaker names with underscores
                        speaker_name = entry['speaker'].replace(' ', '_')
                        rttm_file.write(
                            f"SPEAKER {entry['episode']} 1 {entry['utterance_start']} "
                            f"{entry['duration']} <NA> <NA> {speaker_name} <NA> <NA>\n"
                        )
            print(f"True RTTM file created at {self.true_rttm_path}")
        else:
            print(f"True RTTM file already exists at {self.true_rttm_path}")

    def get_true_rttm_path(self):
        return self.true_rttm_path

if __name__ == "__main__":
    audio_file = "dataset/audio/11.mp3"
    transcript_file = "dataset/transcript/11.json"
    output_dir = "outputs"
    data_loader = DataLoader(audio_file, transcript_file, output_dir)
    data_loader.generate_true_rttm()
    true_rttm_path = data_loader.get_true_rttm_path()
    print(f"True RTTM file path: {true_rttm_path}")