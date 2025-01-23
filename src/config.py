from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    hf_token: str = os.getenv("HF_TOKEN")
    output_dir: str = "outputs"
    dataset_dir: str = "dataset"
    audio_dir: str = os.path.join(dataset_dir, "audio")
    transcript_dir: str = os.path.join(dataset_dir, "transcript")
    true_rttm_dir: str = os.path.join(output_dir, "true_rttm")
    pred_rttm_dir: str = os.path.join(output_dir, "pred_rttm")

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.true_rttm_dir, exist_ok=True)
        os.makedirs(self.pred_rttm_dir, exist_ok=True)

config = Config()