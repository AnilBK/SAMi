import os, sys, random, warnings, importlib.util, gc
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from collections import Counter, defaultdict
import numpy as np
from torch import amp
from torch.amp import GradScaler
import glob
from tqdm import tqdm

DEFAULT_SAMPLE_RATE = 44100
PANN_SAMPLE_RATE = 32000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
NUM_WORKERS = 7
MAX_WINDOW_SECONDS = 1.2

WINDOW_SAMPLES_MAX = int(MAX_WINDOW_SECONDS * PANN_SAMPLE_RATE)

DATASET_DIR = "Dataset_preprocessed"
LOW_SOUND_FILE = "low_sound.txt"
PANN_PATH = "panns/audioset_tagging_cnn"
PANN_CHECKPOINT_PATH = "panns/MobileNetV2_mAP=0.383.pth"

WINDOW_CONFIGS = {
    "Micro": {"duration": 0.6, "hop": 0.1, "threshold": 0.25},
    "Standard": {"duration": 1.0, "hop": 0.3, "threshold": 0.25},
    "Macro": {"duration": 1.2, "hop": 0.5, "threshold": 0.20},
}

WORD_CENTRIC_CONTEXT_SECONDS = 0.6
enable_mixup = False
MIXUP_ALPHA = 0.3
BACKGROUND_NOISE_PROB = 0.8
enable_cnn_unfreezing = True

SHORT_WORD_DURATION_THRESHOLD = 0.2
SHORT_WORD_WINDOW_SECONDS = 0.4

MODEL_CONFIG_LIGHTWEIGHT = {
    "tdnn_dim": 384,
    "pooling_heads": 4,
    "classifier_hidden_dim": 192,
    "classifier_heads": 3,
    "dropout": 0.2
}

LABELING_CONFIG = {
    '1': 0.112, '10': 0.126, '11': 0.285, '12': 0.329, '15': 0.429, '2': 0.161,
    '20': 0.232, '25': 0.241, '3': 0.141, '30': 0.251, '4': 0.147, '5': 0.163,
    '6': 0.159, '7': 0.128, '8': 0.131, '9': 0.155, 'AAJA': 0.203, 'AAJAKO': 0.464,
    'AGADI': 0.286, 'AHILE': 0.247, 'ALARM': 0.144, 'AM': 0.269, 'ANIL': 0.218,
    'ARKO': 0.285, 'BAAR': 0.194, 'BADHAU': 0.205, 'BAJAU': 0.2, 'BAJERA': 0.354,
    'BAJYO': 0.218, 'BALA': 0.202, 'BANAU': 0.192, 'BANDA': 0.167, 'BANDAGARA': 0.284,
    'BATTERY': 0.229, 'BATTI': 0.223, 'BELUKA': 0.249, 'BHANXA': 0.233, 'BHAYO': 0.193,
    'BHOLI': 0.337, 'BIHANA': 0.268, 'BUS': 0.183, 'CALL': 0.18, 'CAMERA': 0.268,
    'CHALAU': 0.22, 'CHALAUXA': 0.292, 'CHORA': 0.209, 'CHORI': 0.232, 'CLOSE': 0.254,
    'DATA': 0.176, 'DEKHAU': 0.208, 'DEUSO': 0.247, 'DEVIL': 0.534, 'DHOKA': 0.217,
    'DIN': 0.177, 'ERESH': 0.47, 'FACEBOOK': 0.217, 'FAN': 0.187, 'FLASHLIGHT': 0.335,
    'GARA': 0.177, 'GARADEU': 0.234, 'GAREYKAI': 0.272, 'GATEY': 0.221, 'GAYO': 0.243,
    'GEET': 0.116, 'GHAM': 0.218, 'GHANTA': 0.218, 'GHATAU': 0.208, 'HATAU': 0.287,
    'HEATER': 0.212, 'HERA': 0.204, 'HIJO': 0.315, 'HO': 0.129, 'HUNEXA': 0.353,
    'HUNXA': 0.451, 'I': 0.234, 'ISHU': 0.493, 'JANE': 0.19, 'K': 0.161, 'KA': 0.174,
    'KAATA': 0.222, 'KAHILE': 0.252, 'KASARI': 0.258, 'KASTO': 0.215, 'KATA': 0.213,
    'KATI': 0.122, 'KHICHDEU': 0.29, 'KHOLA': 0.169, 'KO': 0.167, 'KUN': 0.16,
    'LAGAU': 0.214, 'LAGXA': 0.289, 'LAI': 0.18, 'LIGHT': 0.168, 'LOCK': 0.172,
    'MA': 0.191, 'MANJIL': 0.285, 'MAP': 0.159, 'MARTIN': 0.459, 'MAUSAM': 0.232,
    'MERO': 0.226, 'MESSAGE': 0.258, 'MIN': 0.23, 'MIRA': 0.225, 'MOBILE': 0.257,
    'NATI': 0.232, 'NATINI': 0.372, 'NEPALI': 0.286, 'NIBHAU': 0.208, 'NUMBER': 0.252,
    'OFF': 0.142, 'ON': 0.127, 'PAANI': 0.275, 'PADHA': 0.208, 'PANI': 0.342,
    'PANKHA': 0.276, 'PARCHA': 0.418, 'PARXA': 0.293, 'PHONE': 0.165, 'PHOTO': 0.22,
    'PRADEEP': 0.406, 'PRATISTHA': 0.517, 'RACHANA': 0.54, 'RAKHA': 0.2,
    'RAKHDEU': 0.277, 'RATI': 0.332, 'RISHAV': 0.493, 'ROKA': 0.247, 'SAJHA': 0.241,
    'SAMI': 0.243, 'SAMIKSHYA': 0.445, 'SAMMA': 0.24, 'SAMRIDHI': 0.453,
    'SANJITA': 0.517, 'SARANSH': 0.557, 'SECOND': 0.216, 'SHREYA': 0.58,
    'SHRUTI': 0.464, 'SISAN': 0.459, 'START': 0.258, 'STOP': 0.217, 'SUMINA': 0.277,
    'SUNIL': 0.395, 'THANK': 0.188, 'THYO': 0.303, 'TIME': 0.144, 'TIMER': 0.176,
    'TV': 0.196, 'UTHAU': 0.222, 'VIDEO': 0.242, 'VOLUME': 0.2, 'WEATHER': 0.209,
    'WHERE': 0.311, 'WIFI': 0.271, 'XU': 0.168, 'YOU': 0.224, '__default__': 0.1,
}

def parse_label_file(txt_path):
    labels = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    s, e, w = float(parts[0]), float(parts[1]), parts[2]
                    labels.append((s, e, w))
                except:
                    continue
    return labels


def apply_random_gain(waveform, filename, low_sound_set):
    basename_without_ext = os.path.splitext(os.path.basename(filename))[0]
    if basename_without_ext in low_sound_set:
        gain_db = random.uniform(5, 25)
        waveform = waveform * (10 ** (gain_db / 20))
    return waveform


def get_label_vector_for_window(labels, start_t, end_t, label_map, label_config):
    num_classes = len(label_map)
    label_vector = torch.zeros(num_classes)
    default_threshold = label_config["__default__"]

    for s, e, w in labels:
        actual_word_duration = e - s
        if actual_word_duration < 1e-6: continue
            
        overlap = max(0, min(e, end_t) - max(s, start_t))
        min_overlap_seconds = label_config.get(w, default_threshold)
        
        if overlap > min_overlap_seconds:
            label_idx = label_map.get(w)
            if label_idx is not None:
                presence_score = overlap / actual_word_duration
                label_vector[label_idx] = max(label_vector[label_idx], presence_score)
                
    if torch.sum(label_vector) == 0:
        label_vector[label_map["<SILENCE>"]] = 1.0
        
    return label_vector


def group_txt_files_by_names(directory, names):
    all_files = os.listdir(directory)
    txt_files = {f for f in all_files if f.lower().endswith(".txt")}
    wav_files = {f for f in all_files if f.lower().endswith(".wav")}
    wav_basenames = {os.path.splitext(f)[0] for f in wav_files}
    valid_txts = [f for f in txt_files if os.path.splitext(f)[0] in wav_basenames]
    grouped = defaultdict(list)
    for txt in valid_txts:
        lower_txt = txt.lower()
        matched = False
        for name in names:
            if name.lower() in lower_txt:
                grouped[name].append(txt)
                matched = True
                break
        if not matched:
            grouped["ungrouped"].append(txt)
    return dict(grouped)


def split_speaker_files(grouped_files, val_ratio=0.1):
    train_files = defaultdict(list)
    val_files = defaultdict(list)
    print(f"Splitting files with validation ratio: {val_ratio}")
    for speaker, files in grouped_files.items():
        if not files: continue
        random.shuffle(files)
        val_count = max(1, int(len(files) * val_ratio))
        if len(files) <= val_count:
            val_count = 1 if len(files) > 1 else 0
        val_files[speaker] = files[:val_count]
        train_files[speaker] = files[val_count:]
        print(f"\t{speaker}: Train={len(train_files[speaker])}, Val={len(val_files[speaker])}")
    return dict(train_files), dict(val_files)


def load_low_sound_list(file_path):
    if not os.path.exists(file_path): return set()
    with open(file_path, "r") as f:
        lines = [os.path.splitext(line.strip())[0] for line in f if line.strip()]
    return set(lines)


def build_label_map(dataset_dir, min_count=10):
    print(f"Building label map with minimum count: {min_count}...")
    counter = Counter()
    for fname in os.listdir(dataset_dir):
        if fname.lower().endswith(".txt"):
            path = os.path.join(dataset_dir, fname)
            for _, _, w in parse_label_file(path):
                counter[w] += 1
    print(f"Found {len(counter)} unique labels before filtering.")
    vocab = [word for word, count in counter.items() if count >= min_count]
    print(f"Kept {len(vocab)} labels after filtering.")
    vocab = ["<SILENCE>"] + sorted(vocab)
    print(f"The labels are: {vocab}")
    return {word: i for i, word in enumerate(vocab)}


class PyTorchAudioAugmentations:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.augmentations = [
            {"func": self.apply_gain, "p": 0.6, "params": {"min_gain_db": -8.0, "max_gain_db": 8.0}},
            {"func": self.apply_gaussian_noise, "p": 0.7, "params": {"min_snr_db": 3.0, "max_snr_db": 25.0}},
        ]

    def apply_gain(self, samples, **params):
        gain_db = random.uniform(params["min_gain_db"], params["max_gain_db"])
        return samples * (10 ** (gain_db / 20.0))

    def apply_gaussian_noise(self, samples, **params):
        snr_db = random.uniform(params["min_snr_db"], params["max_snr_db"])
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = torch.mean(samples.pow(2))
        if signal_power < 1e-10: return samples
        noise = torch.randn_like(samples)
        noise_power_current = torch.mean(noise.pow(2))
        if noise_power_current < 1e-10: return samples
        noise_power_target = signal_power / snr_linear
        scale = torch.sqrt(noise_power_target / noise_power_current)
        return samples + (noise * scale)

    def __call__(self, samples):
        augmented_samples = samples
        random.shuffle(self.augmentations)
        for aug in self.augmentations:
            if random.random() < aug["p"]:
                augmented_samples = aug["func"](augmented_samples, **aug["params"])
        return augmented_samples

class ImprovedMultiScaleWordDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        speaker_files,
        label_map,
        label_config,
        low_sound_set,
        jitter_s=0.1,
        augmentations=None,
        is_validation=False,
    ):
        self.dataset_dir = dataset_dir
        self.label_map = label_map
        self.label_config = label_config
        self.silence_idx = self.label_map["<SILENCE>"]
        self.low_sound_set = low_sound_set
        self.jitter_s = jitter_s
        self.augmentations = augmentations
        self.is_validation = is_validation
        
        self.file_list = [
            (file_name, speaker)
            for speaker, files in speaker_files.items()
            for file_name in files
        ]
        self.max_target_samples = int(MAX_WINDOW_SECONDS * PANN_SAMPLE_RATE)
        
        self.label_cache = {}
        self.audio_info_cache = {}

        self.word_windows = self._build_word_inventory()
        self.silence_windows, self.hard_negative_windows = self._build_silence_inventories()
        
        self.noise_segments = self._get_noise_segments()
        if not self.is_validation:
            print(f"Found {len(self.noise_segments)} noise segments for background mixing.")
            print(f"Found {len(self.hard_negative_windows)} hard negative windows.")
        
        self.windows = self._create_full_dataset()
        
        print(f"Created {'validation' if self.is_validation else 'training'} dataset with {len(self.windows)} windows")
        
        if not self.is_validation:
            self.window_labels_for_sampler = self._precalculate_labels_for_sampler()

        self._print_dataset_statistics()
    
    def _get_labels_from_path(self, txt_path):
        if txt_path not in self.label_cache:
            self.label_cache[txt_path] = parse_label_file(txt_path)
        return self.label_cache[txt_path]
    
    def _get_audio_info(self, wav_path):
        if wav_path not in self.audio_info_cache:
            self.audio_info_cache[wav_path] = torchaudio.info(wav_path)
        return self.audio_info_cache[wav_path]

    def _get_noise_segments(self):
        return [window for window in self.silence_windows if window[5] == 'noisy']

    def _mix_with_background_noise(self, clean_waveform, min_snr_db=0, max_snr_db=20):
        if not self.noise_segments:
            return clean_waveform

        noise_info = random.choice(self.noise_segments)
        noise_wav_path, _, noise_start_t, noise_end_t, *_ = noise_info

        try:
            info = self._get_audio_info(noise_wav_path)
            sr = info.sample_rate
            start_sample = int(noise_start_t * sr)
            num_samples = int((noise_end_t - noise_start_t) * sr)
            noise_seg, _ = torchaudio.load(noise_wav_path, frame_offset=start_sample, num_frames=num_samples)
            noise_seg = torch.mean(noise_seg, dim=0, keepdim=True)
        except Exception:
            return clean_waveform

        target_len = clean_waveform.shape[1]
        if noise_seg.shape[1] < target_len:
            noise_seg = noise_seg.repeat(1, (target_len // noise_seg.shape[1]) + 1)
        
        start_idx = random.randint(0, noise_seg.shape[1] - target_len)
        noise_seg = noise_seg[:, start_idx : start_idx + target_len]

        clean_power = torch.mean(clean_waveform.pow(2))
        noise_power = torch.mean(noise_seg.pow(2))

        if clean_power < 1e-10 or noise_power < 1e-10:
            return clean_waveform
        
        snr_db = random.uniform(min_snr_db, max_snr_db)
        snr_linear = 10 ** (snr_db / 10.0)
        
        scale = torch.sqrt(clean_power / (snr_linear * noise_power))
        mixed_waveform = clean_waveform + noise_seg * scale
        
        return torch.clamp(mixed_waveform, -1.0, 1.0)

    def _precalculate_labels_for_sampler(self):
        print("Pre-calculating primary labels for weighted sampler...")
        labels_list = []

        for window_info in tqdm(self.windows, desc="Calculating Sampler Labels"):
            txt_path, start_t, end_t = window_info[1], window_info[2], window_info[3]
            
            labels_from_file = self._get_labels_from_path(txt_path)
            label_vector = get_label_vector_for_window(labels_from_file, start_t, end_t, self.label_map, self.label_config)
            
            present_indices = (label_vector > 0.5).nonzero(as_tuple=False).squeeze(-1)
            non_silence_indices = [idx.item() for idx in present_indices if idx.item() != self.silence_idx]

            primary_label_idx = random.choice(non_silence_indices) if non_silence_indices else self.silence_idx
            labels_list.append(primary_label_idx)
        
        print(f"Finished pre-calculating {len(labels_list)} primary labels for sampler.")
        return labels_list

    def _build_word_inventory(self):
        print("Building word inventory with multi-strategy...")
        all_windows = []
        for file_name, speaker in tqdm(self.file_list, desc="Building Inventory"):
            txt_path, wav_path, info, duration, labels, env = self._get_file_metadata(file_name)
            if not labels: continue
            
            for _, config in WINDOW_CONFIGS.items():
                win_duration, win_hop = config["duration"], config["hop"]
                for start_t in np.arange(0, duration - win_duration + win_hop, win_hop):
                    end_t = start_t + win_duration
                    if any(word for s,e,word in labels if max(0, min(e, end_t) - max(s, start_t)) > 0):
                        all_windows.append((wav_path, txt_path, start_t, end_t, speaker, env, None, None))
            
            if not self.is_validation:
                for s, e, word in labels:
                    if word not in self.label_map or word == "<SILENCE>": continue
                    
                    word_dur = e - s
                    win_duration = min(MAX_WINDOW_SECONDS, word_dur + WORD_CENTRIC_CONTEXT_SECONDS)
                    word_center = s + word_dur / 2
                    
                    start_t = max(0, word_center - win_duration / 2)
                    end_t = min(duration, start_t + win_duration)
                    if end_t - start_t < win_duration:
                        start_t = max(0, end_t - win_duration)
                    all_windows.append((wav_path, txt_path, start_t, end_t, speaker, env, s, e))
                    
                    if word_dur < SHORT_WORD_DURATION_THRESHOLD:
                        win_duration_short = SHORT_WORD_WINDOW_SECONDS
                        start_t_short = max(0, word_center - win_duration_short / 2)
                        end_t_short = min(duration, start_t_short + win_duration_short)
                        if end_t_short - start_t_short < win_duration_short:
                            start_t_short = max(0, end_t_short - win_duration_short)
                        all_windows.append((wav_path, txt_path, start_t_short, end_t_short, speaker, env, s, e))

        return list(dict.fromkeys(all_windows))

    def _build_silence_inventories(self):
        print("Building silence and hard negative inventories...")
        silence_windows = []
        hard_negative_windows = []
        
        for file_name, speaker in self.file_list:
            txt_path, wav_path, info, duration, labels, env = self._get_file_metadata(file_name)
            if not info: continue
            
            for _, config in WINDOW_CONFIGS.items():
                win_duration, win_hop = config["duration"], config["hop"]
                for start_t in np.arange(0, duration - win_duration + win_hop, win_hop):
                    end_t = start_t + win_duration
                    if self._is_true_silence(labels, start_t, end_t):
                        window_info = (wav_path, txt_path, start_t, end_t, speaker, env, None, None)
                        silence_windows.append(window_info)

                        # Check energy to decide if it's a hard negative
                        start_sample = int(start_t * info.sample_rate)
                        num_samples = int(win_duration * info.sample_rate)
                        try:
                            # Load a small chunk to check energy without loading the whole file
                            seg, _ = torchaudio.load(wav_path, frame_offset=start_sample, num_frames=num_samples)
                            energy = torch.mean(seg.pow(2))
                            # Add if energy is between a low and mid threshold (i.e., not pure silence, not loud speech)
                            if 1e-6 < energy < 5e-4 and not self.is_validation:
                                hard_negative_windows.append(window_info)
                        except Exception:
                            continue # Skip if reading fails
                            
        return list(dict.fromkeys(silence_windows)), list(dict.fromkeys(hard_negative_windows))

    def _get_file_metadata(self, file_name):
        txt_path = os.path.join(self.dataset_dir, file_name)
        wav_path = os.path.splitext(txt_path)[0] + ".wav"
        if not os.path.exists(wav_path): return None, None, None, None, None, None
        try:
            info = self._get_audio_info(wav_path)
            duration = info.num_frames / info.sample_rate
            labels = self._get_labels_from_path(txt_path)
            env = "noisy" if "noisy" in file_name.lower() else "clean"
            return txt_path, wav_path, info, duration, labels, env
        except Exception:
            return None, None, None, None, None, None

    def _is_true_silence(self, labels, start_t, end_t):
        return not any(word in self.label_map and word != "<SILENCE>" for s, e, word in labels if max(s, start_t) < min(e, end_t))

    def _create_full_dataset_old(self):
        print("Creating full dataset with hard negatives...")
        # Add hard negatives multiple times to increase their importance
        num_hard_neg_repeats = 2 if not self.is_validation else 0
        all_windows = self.word_windows + self.silence_windows + (self.hard_negative_windows * num_hard_neg_repeats)
        random.shuffle(all_windows)
        return all_windows

    def _create_full_dataset(self):
        print("Creating full dataset with hard negatives...")
        # Add hard negatives multiple times to increase their importance
        num_hard_neg_repeats = 2 if not self.is_validation else 0
        
        if not self.is_validation:
            if len(self.label_map) > 1:
                target_silence_count = len(self.word_windows) // (len(self.label_map) - 1)
            else:
                target_silence_count = len(self.word_windows)
            silence_windows = random.sample(self.silence_windows, min(target_silence_count, len(self.silence_windows))) if self.silence_windows else []
            
            target_hard_count = target_silence_count // 2
            hard_negative_windows = random.sample(self.hard_negative_windows, min(target_hard_count, len(self.hard_negative_windows))) if self.hard_negative_windows else []
        else:
            silence_windows = self.silence_windows
            hard_negative_windows = self.hard_negative_windows
        
        all_windows = self.word_windows + silence_windows + (hard_negative_windows * num_hard_neg_repeats)
        random.shuffle(all_windows)
        return all_windows
        
    def _print_dataset_statistics(self):
        if not self.windows: return
        speech_count = len(self.word_windows)
        silence_count = len(self.silence_windows)
        hard_neg_count = len(self.hard_negative_windows)
        total = len(self.windows)
        print(f"\nDataset Statistics:\n  Total windows: {total}\n  - Speech: {speech_count}\n  - Silence: {silence_count}\n  - Hard Negatives: {hard_neg_count} (added multiple times)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        wav_path, txt_path, win_start_t, win_end_t, speaker, env, word_start_t, word_end_t = self.windows[index]
        
        jitter = random.uniform(-self.jitter_s, self.jitter_s) if not self.is_validation else 0
        win_start_t_jittered = max(0, win_start_t + jitter)
        win_end_t_jittered = win_start_t_jittered + (win_end_t - win_start_t)
        
        labels_from_file = self._get_labels_from_path(txt_path)
        label_vector = get_label_vector_for_window(labels_from_file, win_start_t_jittered, win_end_t_jittered, self.label_map, self.label_config)
        
        try:
            info = self._get_audio_info(wav_path)
            sr = info.sample_rate
            start_sample = int(win_start_t_jittered * sr)
            num_samples = int((win_end_t_jittered - win_start_t_jittered) * sr)
            seg, _ = torchaudio.load(wav_path, frame_offset=start_sample, num_frames=num_samples)
        except Exception:
            return self.__getitem__((index + 1) % len(self))
        
        seg = torch.mean(seg, dim=0, keepdim=True)

        is_noisy_centric_word = (env == 'noisy' and word_start_t is not None and not self.is_validation)
        if is_noisy_centric_word:
            word_start_idx = int(max(0, word_start_t - win_start_t_jittered) * sr)
            word_end_idx = int(max(0, word_end_t - win_start_t_jittered) * sr)
            word_end_idx = min(seg.shape[1], word_end_idx)
            if word_start_idx > 0: seg[:, :word_start_idx] *= 0.1
            if word_end_idx < seg.shape[1]: seg[:, word_end_idx:] *= 0.1

        if not self.is_validation:
            is_clean_word_centric = (env == 'clean' and word_start_t is not None)
            is_clean_silence = (env == 'clean' and word_start_t is None)
            if (is_clean_word_centric or (is_clean_silence and random.random() < 0.2)) and random.random() < BACKGROUND_NOISE_PROB:
                seg = self._mix_with_background_noise(seg)
            seg = apply_random_gain(seg, wav_path, self.low_sound_set)
            if self.augmentations:
                seg = self.augmentations(seg.squeeze(0)).unsqueeze(0)
        
        if seg.shape[1] < self.max_target_samples:
            seg = F.pad(seg, (0, self.max_target_samples - seg.shape[1]))
        seg = seg[:, :self.max_target_samples].squeeze(0)
        
        return seg, label_vector, speaker, env

if not os.path.exists(PANN_PATH):
    raise FileNotFoundError(f"PANN repository not found at {PANN_PATH}.")
sys.path.append(os.path.abspath(PANN_PATH))
sys.path.append(os.path.abspath(os.path.join(PANN_PATH, "pytorch")))
spec = importlib.util.spec_from_file_location("models", os.path.join(PANN_PATH, "pytorch/models.py"))
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.se_block = SEBlock(out_channels)

    def forward(self, x):
        total_padding = self.dilation * (self.kernel_size - 1)
        left_pad = total_padding // 2
        right_pad = total_padding - left_pad
        x = F.pad(x, (left_pad, right_pad))
        x = self.relu(self.bn(self.conv(x)))
        return self.se_block(x)


class GatedAttentionPooling(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())

    def forward(self, x):
        query = torch.mean(x, dim=1, keepdim=True)
        attn_output, _ = self.attn(query, x, x)
        gated = attn_output * self.gate(query)
        mean = gated.squeeze(1)
        std = torch.std(x, dim=1)
        max_ = torch.max(x, dim=1).values
        return torch.cat((mean, std, max_), dim=1)


class AttentionClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_heads=4, dropout=0.15):
        super().__init__()
        self.in_dim = in_dim * 3
        
        self.projection = nn.Linear(in_dim, hidden_dim)
        
        self.attention = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        B, _ = x.shape
        D = x.size(1) // 3
        x_reshaped = x.view(B, 3, D)
        
        projected_x = self.projection(x_reshaped)
        
        attn_out = self.attention(projected_x)
        
        pooled_out = torch.mean(attn_out, dim=1)
        norm_out = self.layer_norm(pooled_out)
        
        return self.fc(norm_out)
    
class SAMIModel(nn.Module):
    def __init__(self, n_mels=64, num_classes=None, config = None, PANN_SAMPLE_RATE=16000, N_FFT=1024):
        super(SAMIModel, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be set")
        if config is None:
            config = {"tdnn_dim": 512, "pooling_heads": 4, "classifier_hidden_dim": 256, 
                      "classifier_heads": 4, "dropout": 0.15}

        self.num_classes = num_classes

        tdnn_dim = config["tdnn_dim"]
        dropout = config["dropout"]

        self.pretrained_cnn = models.MobileNetV2(
            sample_rate=PANN_SAMPLE_RATE, window_size=N_FFT,
            hop_size=256, mel_bins=n_mels, fmin=50, fmax=14000,
            classes_num=527
        )

        self.spec_augment = nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=20,
                                              iid_masks=True, p=0.5),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=12,
                                                   iid_masks=True)
        )

        self.tdnn_proj = nn.Linear(1280, tdnn_dim)

        # Ultra short
        self.tdnn_micro = nn.Sequential(
            TDNNLayer(tdnn_dim, tdnn_dim, kernel_size=1, dilation=1),
            TDNNLayer(tdnn_dim, tdnn_dim, kernel_size=1, dilation=2)
        )

        # Short
        self.tdnn_fast = nn.Sequential(
            TDNNLayer(tdnn_dim, tdnn_dim, kernel_size=2, dilation=1),
            TDNNLayer(tdnn_dim, tdnn_dim, kernel_size=2, dilation=2),
            TDNNLayer(tdnn_dim, tdnn_dim, kernel_size=2, dilation=3)
        )

        # Medium context
        self.tdnn_slow = nn.Sequential(
            TDNNLayer(tdnn_dim, tdnn_dim, kernel_size=3, dilation=1),
            TDNNLayer(tdnn_dim, tdnn_dim, kernel_size=3, dilation=3),
            TDNNLayer(tdnn_dim, tdnn_dim, kernel_size=3, dilation=5)
        )

        # Fusion across branches
        self.tdnn_fusion = nn.Linear(tdnn_dim * 3, tdnn_dim)

        self.layer_norm = nn.LayerNorm(tdnn_dim)
        self.dropout = nn.Dropout(dropout)

        self.pooling = GatedAttentionPooling(tdnn_dim, num_heads=4)

        self.classifier_head = AttentionClassifierHead(
            in_dim=tdnn_dim,
            hidden_dim=config["classifier_hidden_dim"],
            num_classes=num_classes,
            num_heads=config["classifier_heads"],
            dropout=dropout
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if "pretrained_cnn" not in name:
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, waveform):
        x = self.pretrained_cnn.spectrogram_extractor(waveform)
        x = self.pretrained_cnn.logmel_extractor(x)

        if self.training and torch.rand(1) < 0.35:
            x = self.spec_augment(x)

        x = self.pretrained_cnn.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = self.pretrained_cnn.features(x)

        features = torch.mean(x, dim=3).transpose(1, 2)
        tdnn_input = self.tdnn_proj(features)
        tdnn_permuted = tdnn_input.permute(0, 2, 1)

        micro_out = self.tdnn_micro(tdnn_permuted)
        fast_out = self.tdnn_fast(tdnn_permuted)
        slow_out = self.tdnn_slow(tdnn_permuted)

        fused_out = torch.cat((micro_out, fast_out, slow_out), dim=1)
        fused_out = self.tdnn_fusion(fused_out.permute(0, 2, 1)).permute(0, 2, 1)

        tdnn_out = fused_out.permute(0, 2, 1)
        tdnn_out = self.layer_norm(tdnn_out + tdnn_input)
        tdnn_out = self.dropout(tdnn_out)

        pooled = self.pooling(tdnn_out)

        return self.classifier_head(pooled)
    
# | Path      | Frames | Duration |
# | --------- | ------ | -------- |
# | **Micro** | 1      | ~0.016 s |
# | **Fast**  | 9      | ~0.144 s |
# | **Slow**  | 39     | ~0.624 s |

def model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss) if self.reduction == "mean" else torch.sum(F_loss)

class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() / x.size(0) # Per-sample loss, then mean
    
@torch.no_grad()
def validate_model(model, dataloader, criterion):
    model.eval()
    total_loss, total_samples = 0.0, 0
    for waveforms, labels, _, _ in dataloader:
        waveforms, labels = waveforms.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with amp.autocast(device_type="cpu", enabled=(DEVICE=="cuda")):
            preds = model(waveforms)
            loss = criterion(preds, labels)
        total_loss += loss.item() * waveforms.size(0)
        total_samples += waveforms.size(0)
    model.train()
    return total_loss / max(1, total_samples)

@torch.no_grad()
def run_inference_with_confidence_old(model, dataset_dir, label_map_rev, device, step):
    """Enhanced inference that shows confidence scores with progressive thresholds"""
    print(f"\n🕵️‍♂️ STEP {step} - MULTI-SCALE INFERENCE TEST WITH CONFIDENCE")
    print("=" * 60)
    
    def get_inference_configs(step):
        """Progressive thresholds that become stricter as training advances"""
        if step < 10000:
            thresholds = {'short': 0.4, 'medium': 0.35, 'long': 0.3}
            votes_needed = 1
        elif step < 20000:
            thresholds = {'short': 0.55, 'medium': 0.5, 'long': 0.45}
            votes_needed = 1
        elif step < 28000:
            thresholds = {'short': 0.6, 'medium': 0.6, 'long': 0.55}
            votes_needed = 2
        else:
            thresholds = {'short': 0.65, 'medium': 0.65, 'long': 0.6}
            votes_needed = 2

        INFERENCE_CONFIGS = {
            "short": {"duration": 0.6, "hop": 0.2, "threshold": thresholds['short']},
            "medium": {"duration": 1.0, "hop": 0.4, "threshold": thresholds['medium']},
            "long": {"duration": 1.4, "hop": 0.6, "threshold": thresholds['long']},
        }
        return INFERENCE_CONFIGS, votes_needed

    INFERENCE_CONFIGS, VOTES_NEEDED = get_inference_configs(step)
    
    wav_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print("  [Warning] No .wav files found in the dataset directory for inference test.")
        return
    
    test_wav_name = random.choice(wav_files)
    test_wav_path = os.path.join(dataset_dir, test_wav_name)
    test_txt_path = os.path.splitext(test_wav_path)[0] + ".txt"

    ground_truth_words = {w for _,_,w in parse_label_file(test_txt_path) if w in label_map_rev.values()} if os.path.exists(test_txt_path) else set()
    print(f"  🔊 Testing on: {test_wav_name}\n  🎯 Ground Truth Words: {ground_truth_words or 'None'}")
    print(f"  ⚙️  Current thresholds: Short={INFERENCE_CONFIGS['short']['threshold']}, Medium={INFERENCE_CONFIGS['medium']['threshold']}, Long={INFERENCE_CONFIGS['long']['threshold']}")
    print(f"  📊 Votes needed for confirmation: {VOTES_NEEDED}")

    try:
        waveform, orig_sr = torchaudio.load(test_wav_path)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if orig_sr != PANN_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=PANN_SAMPLE_RATE)
        
        original_num_samples = waveform.shape[1]

        max_duration = max(config["duration"] for config in INFERENCE_CONFIGS.values())
        max_window_samples = int(max_duration * PANN_SAMPLE_RATE)
        
        waveform = F.pad(waveform, (0, max_window_samples))

    except Exception as e:
        print(f"  [Error] Could not load audio file: {e}")
        return

    model.eval()
    
    all_scale_detections = defaultdict(list)  # For voting
    all_detections_verbose = defaultdict(lambda: defaultdict(list))  # For confidence analysis
    
    for scale_name, config in INFERENCE_CONFIGS.items():
        window_samples = int(config["duration"] * PANN_SAMPLE_RATE)
        hop_samples = int(config["hop"] * PANN_SAMPLE_RATE)
        
        # Iterate up to the original end of the file
        for start_sample in range(0, original_num_samples, hop_samples):
            # Extract the segment for this window
            segment = waveform[:, start_sample : start_sample + window_samples]
            
            # Pad if necessary to match model input size
            if segment.shape[1] < WINDOW_SAMPLES_MAX:
                segment = F.pad(segment, (0, WINDOW_SAMPLES_MAX - segment.shape[1]))
            
            # Truncate if larger than model's max input
            final_segment = segment[:, :WINDOW_SAMPLES_MAX]

            with amp.autocast(device_type="cpu", enabled=(DEVICE=="cuda")):
                probs = torch.sigmoid(model(final_segment.to(device))).squeeze(0)
            
            for i, prob in enumerate(probs):
                word = label_map_rev.get(i)
                if word and word != "<SILENCE>":
                    prob_val = prob.item()
                    # Store all detections above 0.2 for confidence analysis
                    if prob_val > 0.2:
                        all_detections_verbose[word][scale_name].append(prob_val)
                    # Store detections above threshold for voting
                    word_threshold = config["threshold"] + LABELING_CONFIG.get(word, 0.0) * 0.5  # Scale up for short words
                    if prob_val > word_threshold:
                        all_scale_detections[word].append(scale_name)

    print(f"\n  🔍 Confidence Analysis (all detections > 0.2):")
    if all_detections_verbose:
        for word, scales in sorted(all_detections_verbose.items()):
            scale_confs = []
            for scale_name, confs in scales.items():
                if confs:
                    max_conf = max(confs)
                    scale_confs.append(f"{scale_name}:{max_conf:.3f}")
            
            if scale_confs:
                # Count how many scales pass the actual threshold
                scales_passing_threshold = sum(
                    1 for scale_name, confs in scales.items() 
                    if max(confs) > INFERENCE_CONFIGS[scale_name]["threshold"]
                )
                status = "✅" if scales_passing_threshold >= VOTES_NEEDED else "❌"
                print(f"     {status} {word:15} max_conf: {', '.join(scale_confs)}, scales_passing: {scales_passing_threshold}/{VOTES_NEEDED}")
    else:
        print("     No detections above 0.2 confidence")

    # Apply voting logic to get final predictions
    final_predictions = set()
    for word, scales in all_scale_detections.items():
        # A word is confirmed if it's detected by at least VOTES_NEEDED unique scales
        unique_scales = set(scales)
        if len(unique_scales) >= VOTES_NEEDED:
            final_predictions.add(word)

    print(f"\n  🤖 Model Predictions (Confirmed by >= {VOTES_NEEDED} scales):")
    if final_predictions:
        for word in sorted(list(final_predictions)):
            scales_used = set(all_scale_detections[word])
            print(f"     ✅ {word} (detected by: {', '.join(sorted(scales_used))})")
    else:
        print("     None")
    
    # Compare predictions with ground truth for analysis
    if ground_truth_words:
        correct = ground_truth_words & final_predictions
        hallucinations = final_predictions - ground_truth_words
        missed = ground_truth_words - final_predictions
        
        print(f"\n  --- Analysis ---")
        print(f"  ✅ Correct: {correct or 'None'}")
        print(f"  ❌ Incorrect (Hallucinations): {hallucinations or 'None'}")
        print(f"  🤫 Missed: {missed or 'None'}")
        
        # Calculate metrics
        precision = len(correct) / len(final_predictions) if final_predictions else 0
        recall = len(correct) / len(ground_truth_words) if ground_truth_words else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  📈 Metrics: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    print("=" * 60)
    # Ensure model is returned to training mode
    model.train()
    
    return final_predictions


@torch.no_grad()
def run_inference_with_confidence_new(model, dataset_dir, label_map_rev, device, step):
    """
    Performs and analyzes multi-scale inference on a random file.

    This fixed version uses an adaptive strategy with progressive thresholds,
    a gentler dynamic adjustment for word duration, and an adaptive voting
    requirement to provide a more balanced view of model performance throughout training.
    """
    print(f"\n🕵️‍♂️ STEP {step} - MULTI-SCALE INFERENCE TEST WITH CONFIDENCE")
    print("=" * 60)
    
    def get_inference_configs(step):
        """
        Defines progressively stricter inference parameters based on the training step.
        Returns:
            - INFERENCE_CONFIGS (dict): Windowing configs with thresholds.
            - votes_needed (int): Number of unique scales required for a positive detection.
            - dynamic_thresh_factor (float): A small factor to adjust threshold by word duration.
        """
        if step < 15000:
            thresholds = {'short': 0.45, 'medium': 0.40, 'long': 0.35}
            votes_needed = 1
            dynamic_thresh_factor = 0.20
        elif step < 30000:
            thresholds = {'short': 0.55, 'medium': 0.50, 'long': 0.45}
            votes_needed = 2 
            dynamic_thresh_factor = 0.15
        else: # Late-stage training: focus on precision
            thresholds = {'short': 0.60, 'medium': 0.60, 'long': 0.55}
            votes_needed = 2
            dynamic_thresh_factor = 0.1

        INFERENCE_CONFIGS = {
            "short": {"duration": 0.6, "hop": 0.2, "threshold": thresholds['short']},
            "medium": {"duration": 1.0, "hop": 0.4, "threshold": thresholds['medium']},
            "long": {"duration": 1.4, "hop": 0.6, "threshold": thresholds['long']},
        }
        return INFERENCE_CONFIGS, votes_needed, dynamic_thresh_factor

    INFERENCE_CONFIGS, VOTES_NEEDED, DYNAMIC_THRESH_FACTOR = get_inference_configs(step)
    
    wav_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print("  [Warning] No .wav files found in the dataset directory for inference test.")
        return
    
    test_wav_name = random.choice(wav_files)
    test_wav_path = os.path.join(dataset_dir, test_wav_name)
    test_txt_path = os.path.splitext(test_wav_path)[0] + ".txt"

    ground_truth_words = {w for _,_,w in parse_label_file(test_txt_path) if w in label_map_rev.values()} if os.path.exists(test_txt_path) else set()
    print(f"  🔊 Testing on: {test_wav_name}\n  🎯 Ground Truth Words: {ground_truth_words or 'None'}")
    print(f"  ⚙️  Current thresholds: Short={INFERENCE_CONFIGS['short']['threshold']}, Medium={INFERENCE_CONFIGS['medium']['threshold']}, Long={INFERENCE_CONFIGS['long']['threshold']}")
    print(f"  📊 Votes needed for confirmation: {VOTES_NEEDED}")
    print(f"  ⚖️  Dynamic Threshold Factor: {DYNAMIC_THRESH_FACTOR}")

    try:
        waveform, orig_sr = torchaudio.load(test_wav_path)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if orig_sr != PANN_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=PANN_SAMPLE_RATE)
        
        original_num_samples = waveform.shape[1]
        max_duration = max(config["duration"] for config in INFERENCE_CONFIGS.values())
        max_window_samples = int(max_duration * PANN_SAMPLE_RATE)
        waveform = F.pad(waveform, (0, max_window_samples))
    except Exception as e:
        print(f"  [Error] Could not load audio file: {e}")
        return

    model.eval()
    
    all_scale_detections = defaultdict(list)
    all_detections_verbose = defaultdict(lambda: defaultdict(list))
    
    for scale_name, config in INFERENCE_CONFIGS.items():
        window_samples = int(config["duration"] * PANN_SAMPLE_RATE)
        hop_samples = int(config["hop"] * PANN_SAMPLE_RATE)
        
        for start_sample in range(0, original_num_samples, hop_samples):
            segment = waveform[:, start_sample : start_sample + window_samples]
            
            if segment.shape[1] < WINDOW_SAMPLES_MAX:
                segment = F.pad(segment, (0, WINDOW_SAMPLES_MAX - segment.shape[1]))
            
            final_segment = segment[:, :WINDOW_SAMPLES_MAX]

            with amp.autocast(device_type="cpu", enabled=(DEVICE=="cuda")):
                probs = torch.sigmoid(model(final_segment.to(device))).squeeze(0)
            
            for i, prob in enumerate(probs):
                word = label_map_rev.get(i)
                if word and word != "<SILENCE>":
                    prob_val = prob.item()
                    if prob_val > 0.2:
                        all_detections_verbose[word][scale_name].append(prob_val)
                    
                    # Use the adaptive factor instead of the aggressive hardcoded 0.5
                    word_duration_adjustment = LABELING_CONFIG.get(word, 0.0) * DYNAMIC_THRESH_FACTOR
                    word_threshold = config["threshold"] + word_duration_adjustment
                    
                    if prob_val > word_threshold:
                        all_scale_detections[word].append(scale_name)

    print(f"\n  🔍 Confidence Analysis (all detections > 0.2):")
    if all_detections_verbose:
        for word, scales in sorted(all_detections_verbose.items()):
            scale_confs = []
            for scale_name, confs in scales.items():
                if confs:
                    max_conf = max(confs)
                    scale_confs.append(f"{scale_name}:{max_conf:.3f}")
            
            if scale_confs:
                scales_passing_threshold = sum(
                    1 for scale_name, confs in scales.items() 
                    if max(confs) > (INFERENCE_CONFIGS[scale_name]["threshold"] + LABELING_CONFIG.get(word, 0.0) * DYNAMIC_THRESH_FACTOR)
                )
                status = "✅" if scales_passing_threshold >= VOTES_NEEDED else "❌"
                print(f"     {status} {word:15} max_conf: {', '.join(scale_confs)}, scales_passing: {scales_passing_threshold}/{VOTES_NEEDED}")
    else:
        print("     No detections above 0.2 confidence")

    final_predictions = set()
    for word, scales in all_scale_detections.items():
        if len(set(scales)) >= VOTES_NEEDED:
            final_predictions.add(word)

    print(f"\n  🤖 Model Predictions (Confirmed by >= {VOTES_NEEDED} scales):")
    if final_predictions:
        for word in sorted(list(final_predictions)):
            scales_used = set(all_scale_detections[word])
            print(f"     ✅ {word} (detected by: {', '.join(sorted(scales_used))})")
    else:
        print("     None")
    
    if ground_truth_words:
        correct = ground_truth_words & final_predictions
        hallucinations = final_predictions - ground_truth_words
        missed = ground_truth_words - final_predictions
        
        print(f"\n  --- Analysis ---")
        print(f"  ✅ Correct: {correct or 'None'}")
        print(f"  ❌ Incorrect (Hallucinations): {hallucinations or 'None'}")
        print(f"  🤫 Missed: {missed or 'None'}")
        
        precision = len(correct) / len(final_predictions) if final_predictions else 0
        recall = len(correct) / len(ground_truth_words) if ground_truth_words else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  📈 Metrics: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    print("=" * 60)
    model.train() # Ensure model is returned to training mode

@torch.no_grad()
def run_inference_with_confidence(model, dataset_dir, label_map_rev, device, step):
    print(f"\n🕵️‍♂️ STEP {step} - [ADAPTED] MULTI-SCALE INFERENCE TEST")
    print("=" * 60)
    
    def get_inference_configs(step):
        if step < 15000:
            thresholds = {'short': 0.60, 'medium': 0.55, 'long': 0.50}
            votes_needed = 1
            dynamic_thresh_factor = 0.15
        elif step < 30000:
            thresholds = {'short': 0.75, 'medium': 0.70, 'long': 0.65}
            votes_needed = 2
            dynamic_thresh_factor = 0.10
            : 
            thresholds = {'short': 0.85, 'medium': 0.80, 'long': 0.75}
            votes_needed = 2
            dynamic_thresh_factor = 0.05

        INFERENCE_CONFIGS = {
            "short": {"duration": 0.6, "hop": 0.2, "threshold": thresholds['short']},
            "medium": {"duration": 1.0, "hop": 0.4, "threshold": thresholds['medium']},
            "long": {"duration": 1.4, "hop": 0.6, "threshold": thresholds['long']},
        }
        return INFERENCE_CONFIGS, votes_needed, dynamic_thresh_factor

    INFERENCE_CONFIGS, VOTES_NEEDED, DYNAMIC_THRESH_FACTOR = get_inference_configs(step)
    
    wav_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print("  [Warning] No .wav files found for inference test.")
        return
    
    test_wav_name = random.choice(wav_files)
    test_wav_path = os.path.join(dataset_dir, test_wav_name)
    test_txt_path = os.path.splitext(test_wav_path)[0] + ".txt"

    ground_truth_words = {w for _,_,w in parse_label_file(test_txt_path) if w in label_map_rev.values()} if os.path.exists(test_txt_path) else set()
    print(f"  🔊 Testing on: {test_wav_name}\n  🎯 Ground Truth Words: {ground_truth_words or 'None'}")
    print(f"  ⚙️  Current thresholds: Short={INFERENCE_CONFIGS['short']['threshold']:.2f}, Medium={INFERENCE_CONFIGS['medium']['threshold']:.2f}, Long={INFERENCE_CONFIGS['long']['threshold']:.2f}")
    print(f"  📊 Votes needed for confirmation: {VOTES_NEEDED}")
    print(f"  ⚖️  Dynamic Threshold Factor: {DYNAMIC_THRESH_FACTOR:.2f}")

    try:
        waveform, orig_sr = torchaudio.load(test_wav_path)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if orig_sr != PANN_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=PANN_SAMPLE_RATE)
        
        original_num_samples = waveform.shape[1]
        max_duration = max(config["duration"] for config in INFERENCE_CONFIGS.values())
        max_window_samples = int(max_duration * PANN_SAMPLE_RATE)
        waveform = F.pad(waveform, (0, max_window_samples))
    except Exception as e:
        print(f"  [Error] Could not load audio file: {e}")
        return

    model.eval()
    
    all_scale_detections = defaultdict(list)
    all_detections_verbose = defaultdict(lambda: defaultdict(list))
    
    for scale_name, config in INFERENCE_CONFIGS.items():
        window_samples = int(config["duration"] * PANN_SAMPLE_RATE)
        hop_samples = int(config["hop"] * PANN_SAMPLE_RATE)
        
        for start_sample in range(0, original_num_samples, hop_samples):
            segment = waveform[:, start_sample : start_sample + window_samples]
            
            if segment.shape[1] < WINDOW_SAMPLES_MAX:
                segment = F.pad(segment, (0, WINDOW_SAMPLES_MAX - segment.shape[1]))
            
            final_segment = segment[:, :WINDOW_SAMPLES_MAX]

            with amp.autocast(device_type="cpu", enabled=(device=="cuda")):
                probs = torch.sigmoid(model(final_segment.to(device))).squeeze(0)
            
            for i, prob in enumerate(probs):
                word = label_map_rev.get(i)
                if word and word != "<SILENCE>":
                    prob_val = prob.item()
                    if prob_val > 0.2: 
                        all_detections_verbose[word][scale_name].append(prob_val)
                    
                    word_duration_adjustment = LABELING_CONFIG.get(word, 0.0) * DYNAMIC_THRESH_FACTOR
                    word_threshold = config["threshold"] + word_duration_adjustment
                    
                    if prob_val > word_threshold:
                        all_scale_detections[word].append(scale_name)

    print(f"\n  🔍 Confidence Analysis (all detections > 0.2):")
    if all_detections_verbose:
        for word, scales in sorted(all_detections_verbose.items()):
            scale_confs = []
            for scale_name, confs in scales.items():
                if confs:
                    max_conf = max(confs)
                    scale_confs.append(f"{scale_name}:{max_conf:.3f}")
            
            if scale_confs:
                scales_passing_threshold = sum(
                    1 for scale_name, confs in scales.items() 
                    if max(confs) > (INFERENCE_CONFIGS[scale_name]["threshold"] + LABELING_CONFIG.get(word, 0.0) * DYNAMIC_THRESH_FACTOR)
                )
                status = "✅" if scales_passing_threshold >= VOTES_NEEDED else "❌"
                print(f"     {status} {word:15} max_conf: {', '.join(scale_confs)}, scales_passing: {scales_passing_threshold}/{VOTES_NEEDED}")
    else:
        print("     No detections above 0.2 confidence")

    final_predictions = set()
    for word, scales in all_scale_detections.items():
        if len(set(scales)) >= VOTES_NEEDED:
            final_predictions.add(word)

    print(f"\n  🤖 Model Predictions (Confirmed by >= {VOTES_NEEDED} scales):")
    if final_predictions:
        for word in sorted(list(final_predictions)):
            scales_used = set(all_scale_detections[word])
            print(f"     ✅ {word} (detected by: {', '.join(sorted(scales_used))})")
    else:
        print("     None")
    
    if ground_truth_words:
        correct = ground_truth_words & final_predictions
        hallucinations = final_predictions - ground_truth_words
        missed = ground_truth_words - final_predictions
        
        print(f"\n  --- Analysis ---")
        print(f"  ✅ Correct: {correct or 'None'}")
        print(f"  ❌ Incorrect (Hallucinations): {hallucinations or 'None'}")
        print(f"  🤫 Missed: {missed or 'None'}")
        
        precision = len(correct) / len(final_predictions) if final_predictions else 0
        recall = len(correct) / len(ground_truth_words) if ground_truth_words else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  📈 Metrics: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    print("=" * 60)
    model.train()

@torch.no_grad()
def run_inference_on_random_file(model, dataset_dir, label_map_rev, device, step):
    print(f"\n🕵️‍♂️ STEP {step} - MULTI-SCALE INFERENCE TEST")
    print("=" * 60)
    
    INFERENCE_CONFIGS = {
        "short": {"duration": 0.6, "hop": 0.2, "threshold": 0.65},
        "medium": {"duration": 1.0, "hop": 0.4, "threshold": 0.65},
        "long": {"duration": 1.4, "hop": 0.6, "threshold": 0.60},
    }
    VOTES_NEEDED = 2
    
    wav_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print("  [Warning] No .wav files found in the dataset directory for inference test.")
        return

    test_wav_name = random.choice(wav_files)
    test_wav_path = os.path.join(dataset_dir, test_wav_name)
    test_txt_path = os.path.splitext(test_wav_path)[0] + ".txt"

    ground_truth_words = {w for _,_,w in parse_label_file(test_txt_path) if w in label_map_rev.values()} if os.path.exists(test_txt_path) else set()
    print(f"  🔊 Testing on: {test_wav_name}\n  🎯 Ground Truth Words: {ground_truth_words or 'None'}")

    try:
        waveform, orig_sr = torchaudio.load(test_wav_path)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if orig_sr != PANN_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=PANN_SAMPLE_RATE)
        
        original_num_samples = waveform.shape[1]

        max_duration = max(config["duration"] for config in INFERENCE_CONFIGS.values())
        max_window_samples = int(max_duration * PANN_SAMPLE_RATE)
        
        waveform = F.pad(waveform, (0, max_window_samples))

    except Exception as e:
        print(f"  [Error] Could not load audio file: {e}")
        return

    model.eval()
    all_scale_detections = defaultdict(list)
    
    for scale_name, config in INFERENCE_CONFIGS.items():
        window_samples = int(config["duration"] * PANN_SAMPLE_RATE)
        hop_samples = int(config["hop"] * PANN_SAMPLE_RATE)
        
        for start_sample in range(0, original_num_samples, hop_samples):
            segment = waveform[:, start_sample : start_sample + window_samples]
            
            if segment.shape[1] < WINDOW_SAMPLES_MAX:
                segment = F.pad(segment, (0, WINDOW_SAMPLES_MAX - segment.shape[1]))
            
            # In case a window is larger than the model's max input, truncate it.
            final_segment = segment[:, :WINDOW_SAMPLES_MAX]

            with amp.autocast("cpu", enabled=(DEVICE=="cuda")):
                probs = torch.sigmoid(model(final_segment.to(device))).squeeze(0)
            
            for i, prob in enumerate(probs):
                if prob.item() > config["threshold"]:
                    word = label_map_rev.get(i)
                    if word and word != "<SILENCE>":
                        all_scale_detections[word].append(scale_name)

    # Apply voting logic to get final predictions
    final_predictions = set()
    for word, scales in all_scale_detections.items():
        # A word is confirmed if it's detected by at least VOTES_NEEDED unique scales
        if len(set(scales)) >= VOTES_NEEDED:
            final_predictions.add(word)

    print(f"\n  🤖 Model Predictions (Confirmed by >= {VOTES_NEEDED} scales):")
    if final_predictions:
        print(f"     {', '.join(sorted(list(final_predictions)))}")
    else:
        print("     None")
    
    if ground_truth_words:
        correct = ground_truth_words & final_predictions
        hallucinations = final_predictions - ground_truth_words
        missed = ground_truth_words - final_predictions
        
        print(f"\n  --- Analysis ---")
        print(f"  ✅ Correct: {correct or 'None'}")
        print(f"  ❌ Incorrect (Hallucinations): {hallucinations or 'None'}")
        print(f"  🤫 Missed: {missed or 'None'}")

    print("=" * 60)
    # Ensure model is returned to training mode
    model.train()

def train_model(model, train_dataloader, val_dataloader, max_steps=60000, lr=5e-4, unfreeze_schedule=None, initial_step=0, val_check_interval=1000, label_map_rev=None):
    global enable_mixup
    log_file_path = "training_log.csv"
    if enable_cnn_unfreezing and unfreeze_schedule is None: unfreeze_schedule = {5000: 2, 10000: 4, 20000: 6}
    for param in model.pretrained_cnn.parameters(): param.requires_grad = False
    model.pretrained_cnn.eval()

    with open(log_file_path, "a", buffering=1) as log_file:
        if not os.path.exists(log_file_path) or initial_step == 0: log_file.write("step,train_loss,val_loss,learning_rate\n")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=max_steps, pct_start=0.1)

        # criterion = FocalLoss(alpha=0.8, gamma=2.0)
        criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05)

        scaler = GradScaler(device=DEVICE if DEVICE == "cuda" else "cpu", enabled=(DEVICE == "cuda"))

        
        print(f"Starting training for {max_steps} steps, resuming from step {initial_step}...")
        current_step, total_loss, total_samples = initial_step, 0.0, 0
        train_data_iterator = iter(train_dataloader)
        
        with tqdm(initial=initial_step, total=max_steps, desc="Training") as pbar:
            while current_step < max_steps:
                if enable_cnn_unfreezing and current_step in unfreeze_schedule:
                    num_blocks = unfreeze_schedule[current_step]
                    pbar.write(f"\nSTEP {current_step}: Unfreezing last {num_blocks} CNN blocks.")
                    model.pretrained_cnn.train()
                    for i in range(len(model.pretrained_cnn.features) - num_blocks, len(model.pretrained_cnn.features)):
                        for param in model.pretrained_cnn.features[i].parameters(): param.requires_grad = True

                try: waveforms, labels, _, _ = next(train_data_iterator)
                except StopIteration: train_data_iterator = iter(train_dataloader); continue

                waveforms, labels = waveforms.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                
                if enable_mixup and model.training and random.random() < 0.75:
                    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    rand_index = torch.randperm(waveforms.size(0)).to(DEVICE, non_blocking=True)
                    mixed_waveforms = lam * waveforms + (1 - lam) * waveforms[rand_index, :]
                    mixed_labels = lam * labels + (1 - lam) * labels[rand_index, :]
                    waveforms = mixed_waveforms
                    labels = mixed_labels
                
                labels = labels * 0.95 + 0.1 / model.num_classes
                
                optimizer.zero_grad(set_to_none=True)
    
                with amp.autocast("cpu", enabled=(DEVICE=="cuda")):
                    loss = criterion(model(waveforms), labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                total_samples += 1
                current_step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{total_loss/total_samples:.5f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if current_step % val_check_interval == 0:
                    avg_train_loss = total_loss / total_samples
                    val_loss = validate_model(model, val_dataloader, criterion)
                    lr_val = optimizer.param_groups[0]['lr']
                    pbar.write(f"[Step {current_step}] Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {lr_val:.6f}")
                    log_file.write(f"{current_step},{avg_train_loss:.6f},{val_loss:.6f},{lr_val:.8f}\n")
                    
                    # run every 1000 steps.
                    if current_step % (val_check_interval) == 0:
                        for i in range(5):
                            run_inference_with_confidence(model, DATASET_DIR, label_map_rev, DEVICE, current_step)
                    
                    total_loss, total_samples = 0.0, 0

                if current_step > 1000 and not enable_mixup:
                    enable_mixup = True

                if current_step > 0 and current_step % 1000 == 0:
                    torch.save(model.state_dict(), f"sami_tdnn_step_{current_step}.pth")

        print("Training complete.")
        torch.save(model.state_dict(), f"sami_tdnn_final_step_{current_step}.pth")

def create_balanced_sampler(dataset):
    print("Creating sampler weights...")
    labels = dataset.window_labels_for_sampler
    class_counts = torch.bincount(torch.tensor(labels), minlength=len(dataset.label_map))
    weights = torch.zeros_like(class_counts, dtype=torch.float)
    weights[class_counts > 0] = 1.0 / class_counts[class_counts > 0]
    sample_weights = weights[torch.tensor(labels)]
    print("Sampler weights created.")
    print("Class counts:", class_counts)
    for idx, count in enumerate(class_counts):
        word = {v: k for k, v in label_map.items()}.get(idx, "Unknown")
        print(f"{word}: {count}")
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

if __name__ == "__main__":
    speaker_names = ["mira", "anup", "igyan", "eresh", "ishu", "manjil", "martin", "munal", "nisha", "samridhi", "sanju", "shreya", "sisan", "sumi", "utsav", "r_", "anil_"]
    
    grouped = group_txt_files_by_names(DATASET_DIR, speaker_names)
    low_sound_set = load_low_sound_list(LOW_SOUND_FILE)
    train_files, val_files = split_speaker_files(grouped, val_ratio=0.1)
    label_map = build_label_map(DATASET_DIR, min_count=10)
    num_classes = len(label_map)
    print(f"\nNumber of classes: {num_classes}")

    train_augmentations = PyTorchAudioAugmentations(sample_rate=PANN_SAMPLE_RATE)

    print("\n=== CREATING TRAINING DATASET ===")
    train_dataset = ImprovedMultiScaleWordDataset(
        DATASET_DIR, train_files, label_map, LABELING_CONFIG, low_sound_set,
        jitter_s=0.1, augmentations=train_augmentations, is_validation=False)
    
    print("\n=== CREATING VALIDATION DATASET ===")
    val_dataset = ImprovedMultiScaleWordDataset(
        DATASET_DIR, val_files, label_map, LABELING_CONFIG, low_sound_set, is_validation=True)
    
    train_sampler = create_balanced_sampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, num_workers=2, pin_memory=(DEVICE == "cuda"), shuffle=False)
    
    print("-" * 50 + f"\nTotal training windows: {len(train_dataset)}\nTotal validation windows: {len(val_dataset)}\nSteps per epoch: ~{len(train_dataloader)}\n" + "-" * 50)
    
    '''
    sami_model = SAMIModel(n_mels=N_MELS, tdnn_dim=512, num_classes=num_classes, dropout=0.15).to(DEVICE)
    Model Summary: SAMIModel
    Total parameters: 13,613,165
    Trainable parameters: 12,529,709
    Non-trainable parameters: 1,083,456
    '''
    sami_model = SAMIModel(n_mels=N_MELS, num_classes=num_classes, config = MODEL_CONFIG_LIGHTWEIGHT).to(DEVICE)
    '''
    Model Summary: SAMIModel
    Total parameters: 10,043,469
    Trainable parameters: 8,960,013
    Non-trainable parameters: 1,083,456
    Pretrained PANN weights loaded successfully.
    '''
    model_summary(sami_model)

    if os.path.exists(PANN_CHECKPOINT_PATH):
        try:
            pretrained_state_dict = torch.load(PANN_CHECKPOINT_PATH, map_location="cpu")["model"]
            sami_model.pretrained_cnn.load_state_dict(pretrained_state_dict)
            print("Pretrained PANN weights loaded successfully.")
        except Exception as e: warnings.warn(f"Could not load PANN checkpoint: {e}. Training from scratch.")
    else: warnings.warn(f"PANN checkpoint not found. Training from scratch.")

    initial_step = 0
    all_checkpoints = sorted(glob.glob("sami_tdnn_step_*.pth"), key=os.path.getmtime, reverse=True)
    if all_checkpoints:
        latest_checkpoint_path = all_checkpoints[0]
        try:
            initial_step = int(os.path.basename(latest_checkpoint_path).split("_")[-1].replace(".pth", ""))
            print(f"Loading latest SAMI checkpoint: {latest_checkpoint_path} (resuming from step {initial_step})")
            sami_model.load_state_dict(torch.load(latest_checkpoint_path, map_location=DEVICE))
        except Exception as e: warnings.warn(f"Could not load SAMI checkpoint: {e}. Starting from step 0."); initial_step = 0
    else: print("No previous SAMI checkpoint found. Starting from step 0.")

    label_map_rev = {v: k for k, v in label_map.items()}

    train_model(
        sami_model, train_dataloader, val_dataloader,
        max_steps=50000, lr=5e-4, initial_step=initial_step,
        unfreeze_schedule={1000: 2, 3000: 4, 6000: 8, 12000: 12},
        val_check_interval=2000, label_map_rev=label_map_rev)