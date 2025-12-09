import os, sys, glob, importlib.util, time, threading, queue, warnings
from collections import Counter, defaultdict, deque
from datetime import datetime
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pyaudio
import pygame
import cv2

try:
    from moviepy.editor import VideoFileClip

    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

warnings.filterwarnings("ignore", category=UserWarning, module="pyaudio")

DEFAULT_SAMPLE_RATE = 44100
PANN_SAMPLE_RATE = 32000
N_MELS = 64
N_FFT = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WINDOW_SECONDS = 1.2
WINDOW_SAMPLES_MAX = int(MAX_WINDOW_SECONDS * PANN_SAMPLE_RATE)
WINDOW_CONFIGS = {
    "Micro": {"duration": 0.6},
    "Standard": {"duration": 1.0},
    "Macro": {"duration": 1.2},
}
INFERENCE_WINDOW_DURATIONS_S = sorted([c["duration"] for c in WINDOW_CONFIGS.values()])
HOP_SECONDS_FOR_INFERENCE = 0.2
HOP_SAMPLES = int(HOP_SECONDS_FOR_INFERENCE * DEFAULT_SAMPLE_RATE)
CHUNK_SECONDS = 0.1
CHUNK_SIZE = int(CHUNK_SECONDS * DEFAULT_SAMPLE_RATE)
FORMAT, CHANNELS = pyaudio.paInt16, 1
DATASET_DIR = "Dataset_preprocessed"
PANN_PATH = "panns/audioset_tagging_cnn"

SAMPLE_VIDEO_PATH = os.path.join("assets", "tv.mp4")
SAMPLE_VIDEO_AUDIO_PATH = os.path.join("assets", "tv.wav")
MUSIC_FILE_PATH = os.path.join("assets", "sumi_music.wav")

WEATHER_CACHE_FILE = "weather.txt"

WEAK_DETECTION_THRESHOLD = 0.45
WEAK_DETECTION_WINDOW = 5
WEAK_DETECTION_FRAMES = 3
WEAK_DETECTION_AVG_PROB = 0.60

MANUAL_THRESHOLDS = {
    "TV": (0.25, 0.45),
    "ON": (0.25, 0.40),
    "OFF": (0.25, 0.40),
    "FAN": (0.30, 0.45),
    "1": (0.20, 0.35),
    "6": (0.20, 0.35),
    "10": (0.20, 0.35),
    "KATI": (0.30, 0.45),
    "BAJYO": (0.30, 0.45),
    "BATTI": (0.35, 0.50),
    "RAKHA": (0.45, 0.40),

    # "GEET": (0.35, 0.35),
    # "DHOKA": (0.15, 0.20),
}

MODEL_CONFIG_LIGHTWEIGHT = {
    "tdnn_dim": 384,
    "pooling_heads": 4,
    "classifier_hidden_dim": 192,
    "classifier_heads": 3,
    "dropout": 0.2,
}

WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 500
GAME_UI_WIDTH = 700
INFO_PANEL_WIDTH = WINDOW_WIDTH - GAME_UI_WIDTH
TV_RECT_X, TV_RECT_Y = 300, 340
TV_RECT_W, TV_RECT_H = 160, 120

GRAY, BLACK, WHITE, BLUE = (200, 200, 200), (0, 0, 0), (255, 255, 255), (50, 50, 200)

WEATHER_SOUND_MAP = {
    "Sunny": os.path.join("assets", "Responses", "Gham_lageko_xa.wav"),
    "Cloudy": os.path.join("assets", "Responses", "Badal_lageko_xa.wav"),
    "Rainy": os.path.join("assets", "Responses", "paani_pareko_xa.wav"),
    "Foggy": os.path.join("assets", "Responses", "hussu_lageko_xa.wav"),
    "Snowy": os.path.join("assets", "Responses", "hussu_lageko_xa.wav"),
    "Thunderstorm": os.path.join("assets", "Responses", "paani_pareko_xa.wav"),
    "Normal": os.path.join("assets", "Responses", "Gham_lageko_xa.wav"),
}


def get_weather_description(code):
    if code == 0:
        return "Sunny"
    if code in [1, 2, 3]:
        return "Cloudy"
    if code in [45, 48]:
        return "Foggy"
    if code in [51, 53, 55]:
        return "Drizzle"
    if code in [61, 63, 65, 80, 81, 82]:
        return "Rainy"
    if code in [71, 73, 75]:
        return "Snowy"
    if code in [95, 96, 99]:
        return "Thunderstorm"
    return "Normal"


def get_weather_in_kathmandu_description(log_queue):
    log_queue.put("Attempting to fetch live weather data for Kathmandu...")
    url = "https://api.open-meteo.com/v1/forecast?latitude=27.7172&longitude=85.3240&current_weather=true"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        weather = response.json()["current_weather"]
        description = get_weather_description(weather["weathercode"])
        log_queue.put(
            f"--- Live Weather Update: {weather['temperature']}°C, {description} ---"
        )
        try:
            with open(WEATHER_CACHE_FILE, "w") as f:
                f.write(description)
        except IOError:
            pass
        return description
    except requests.RequestException as e:
        log_queue.put(f"Error fetching live weather: {e}")
        try:
            with open(WEATHER_CACHE_FILE, "r") as f:
                return f.read().strip() or "Weather Unknown"
        except (FileNotFoundError, IOError):
            return "Weather Unknown"


class GameState:
    def __init__(self, max_logs=50, log_queue=None):
        self.door_state, self.light_state, self.tv_state = "closed", "on", "off"
        self.show_time, self.show_weather = False, False
        self.current_weather_description, self.log_messages, self.latest_predictions = (
            "",
            [],
            [],
        )
        self.silence_prob, self.threshold = 0.0, 0.0
        self._last_logged_label, self._last_logged_ts, self._last_no_confident_ts = (
            None,
            0.0,
            0.0,
        )
        self.max_logs, self.lock, self.log_queue = (
            max_logs,
            threading.RLock(),
            log_queue,
        )
        self.toast_message, self.toast_start_time, self.toast_time = None, None, 3
        self.fan_images = {
            "off": pygame.image.load(os.path.join("assets", "fan_1.png")),
            "on": [
                pygame.image.load(os.path.join("assets", "fan_1.png")),
                pygame.image.load(os.path.join("assets", "fan_2.png")),
            ],
        }
        self.fan_state, self.fan_animation_index = "off", 0
        self.last_fan_toggle_time, self.play_music, self.paused = (
            time.time(),
            False,
            False,
        )

        self.system_busy = False
        self.playing_video = False
        self.audio_queue_ref = None

    def add_log(self, message, also_print=False):
        if also_print and self.log_queue:
            self.log_queue.put(message)
        with self.lock:
            self.log_messages.append(message)
            if len(self.log_messages) > self.max_logs:
                self.log_messages.pop(0)

    def get_logs(self, limit=15):
        with self.lock:
            return self.log_messages[-limit:]

    def set_predictions(self, predictions, threshold, silence_prob):
        to_log, now = None, time.time()
        with self.lock:
            self.latest_predictions, self.threshold, self.silence_prob = (
                predictions[:],
                threshold,
                silence_prob,
            )
            if predictions:
                top_label, top_prob = predictions[0]
                if (
                    top_label != self._last_logged_label
                    or now - self._last_logged_ts > 3
                ):
                    (
                        to_log,
                        self._last_logged_label,
                        self._last_logged_ts,
                        self._last_no_confident_ts,
                    ) = (f"Heard: {top_label} ({top_prob:.2f})", top_label, now, 0.0)
            elif now - self._last_no_confident_ts > 5:
                to_log, self._last_no_confident_ts, self._last_logged_label = (
                    "Listening...",
                    now,
                    None,
                )
        if to_log:
            self.add_log(to_log, also_print=False)

    def get_predictions(self):
        with self.lock:
            return self.latest_predictions[:], self.threshold, self.silence_prob

    def get_states(self):
        with self.lock:
            return (
                self.door_state,
                self.light_state,
                self.tv_state,
                self.show_time,
                self.show_weather,
                self.current_weather_description,
            )

    def set_door_state(self, state):
        with self.lock:
            if self.door_state != state.lower():
                self.door_state = state.lower()
                self.add_log(f"Door {self.door_state.upper()}", True)

    def toggle_light(self):
        with self.lock:
            self.light_state = "off" if self.light_state == "on" else "on"
            self.add_log(f"Light {self.light_state.upper()}", True)

    def set_tv_state(self, state):
        with self.lock:
            self.tv_state = state.lower()
            self.add_log(f"TV {self.tv_state.upper()}", True)

            if self.tv_state == "on":
                self.start_video_playback()
            else:
                self.stop_video_playback()

    def start_video_playback(self):
        if self.playing_video:
            return
        if os.path.exists(SAMPLE_VIDEO_PATH):
            self.playing_video = True
            self.system_busy = True
            self.add_log("Starting Video Playback (Window + Audio)...", True)
            threading.Thread(target=self._video_playback_worker, daemon=True).start()
        else:
            self.add_log(f"Error: Video file not found at {SAMPLE_VIDEO_PATH}", True)

    def _video_playback_worker(self):
        """Plays video using OpenCV in a new window and audio using PyGame."""
        audio_ready = False
        if os.path.exists(SAMPLE_VIDEO_AUDIO_PATH):
            audio_ready = True
        elif HAS_MOVIEPY:
            try:
                self.add_log("Extracting audio from video...", False)
                clip = VideoFileClip(SAMPLE_VIDEO_PATH)
                clip.audio.write_audiofile(
                    SAMPLE_VIDEO_AUDIO_PATH, logger=None, verbose=False
                )
                clip.close()
                audio_ready = True
            except Exception as e:
                self.add_log(f"Audio extraction failed: {e}", True)

        cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)
        if not cap.isOpened():
            self.stop_video_playback()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        frame_delay = 1.0 / fps

        window_name = "SAMI TV Output"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)

        if audio_ready:
            try:
                pygame.mixer.music.load(SAMPLE_VIDEO_AUDIO_PATH)
                pygame.mixer.music.play()
            except Exception as e:
                self.add_log(f"Audio playback failed: {e}", True)

        while self.playing_video and cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                pass

            elapsed = time.time() - start_time
            wait_time = frame_delay - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        cap.release()
        pygame.mixer.music.stop()
        cv2.destroyAllWindows()
        self.stop_video_playback()

    def stop_video_playback(self):
        self.playing_video = False
        self.system_busy = False
        with self.lock:
            self.tv_state = "off"
        self.add_log("Video Playback Finished.", True)
        if self.audio_queue_ref:
            with self.audio_queue_ref.mutex:
                self.audio_queue_ref.queue.clear()
            self.add_log("Buffer cleared, listening resumed.", False)

    def toggle_music(self, state):
        if state:
            if os.path.exists(MUSIC_FILE_PATH):
                self.play_music = True
                self.system_busy = True
                self.add_log(f"Playing Music: {MUSIC_FILE_PATH}", True)
                threading.Thread(
                    target=self._music_playback_worker, daemon=True
                ).start()
            else:
                self.add_log(f"Music file not found: {MUSIC_FILE_PATH}", True)
        else:
            self.play_music = False
            pygame.mixer.music.stop()
            self.system_busy = False
            self.add_log("Music Stopped.", True)

    def _music_playback_worker(self):
        try:
            pygame.mixer.music.load(MUSIC_FILE_PATH)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self.play_music:
                time.sleep(0.5)
        except Exception as e:
            self.add_log(f"Music error: {e}", True)
        finally:
            self.play_music = False
            self.system_busy = False
            if self.audio_queue_ref:
                with self.audio_queue_ref.mutex:
                    self.audio_queue_ref.queue.clear()
            self.add_log("Music finished, listening resumed.", True)

    def toggle_time_display(self):
        with self.lock:
            self.show_time = not self.show_time
            self.add_log(
                f"Time display {'shown' if self.show_time else 'hidden'}", True
            )

        if self.show_time:
            now = datetime.now()
            hour_12 = now.hour % 12
            if hour_12 == 0:
                hour_12 = 12

            target_hours = [1, 2, 3, 4, 10, 11, 12]

            if hour_12 in target_hours:
                audio_filename = f"{hour_12}_bajyo.wav"
                sound_path = os.path.join("assets", "Responses", audio_filename)

                if os.path.exists(sound_path):
                    self.add_log(f"Announcing time: {hour_12}...", True)
                    threading.Thread(
                        target=self._play_audio_blocking, args=(sound_path,)
                    ).start()
                else:
                    self.add_log(
                        f"Audio file missing for time: {audio_filename}", False
                    )

    def set_weather_info(self, description):
        with self.lock:
            self.show_weather = True
            self.current_weather_description = description
            self.add_log(f"Weather: {description}", True)
            sound_path = WEATHER_SOUND_MAP.get(description)
            if sound_path and os.path.exists(sound_path):
                threading.Thread(
                    target=self._play_audio_blocking, args=(sound_path,)
                ).start()
            threading.Timer(10.0, self.hide_weather).start()

    def _play_audio_blocking(self, sound_path):
        try:
            self.system_busy = True
            sound = pygame.mixer.Sound(sound_path)
            length = sound.get_length()
            sound.play()
            self.add_log(f"Speaking... ({length:.1f}s)", False)
            time.sleep(length + 0.5)
        except Exception as e:
            self.add_log(f"Audio Error: {e}", True)
        finally:
            self.system_busy = False
            if self.audio_queue_ref:
                with self.audio_queue_ref.mutex:
                    self.audio_queue_ref.queue.clear()
                self.add_log("Microphone buffer cleared.", False)

    def call_person(self, person_name):
        self.show_toast(f"Calling {person_name}...", 5)

    def hide_weather(self):
        with self.lock:
            self.show_weather = False
            self.add_log("Weather display hidden", True)

    def show_toast(self, message, duration=3):
        self.toast_message, self.toast_start_time, self.toast_time = (
            message,
            time.time(),
            duration,
        )

    def draw_toast(self, screen):
        if (
            self.toast_message
            and self.toast_start_time
            and time.time() - self.toast_start_time < self.toast_time
        ):
            font = pygame.font.Font(None, 36)
            text_surface = font.render(self.toast_message, True, WHITE)
            text_rect = text_surface.get_rect(
                center=(screen.get_width() // 2, screen.get_height() // 2)
            )
            pygame.draw.rect(screen, BLACK, text_rect.inflate(20, 20))
            screen.blit(text_surface, text_rect)
        else:
            self.toast_message, self.toast_start_time = None, None

    def toggle_fan(self, p_toggle):
        self.fan_state = "on" if p_toggle else "off"

    def create_alarm(self):
        self.show_toast("Added alarm for tomorrow morning.", 4)

    def create_timer(self, second):
        self.show_toast(f"Timer set for {second} seconds.", second)

    def draw_fan(self, screen):
        if self.fan_state == "off":
            fan_image = self.fan_images["off"]
        else:
            if time.time() - self.last_fan_toggle_time > 0.5:
                self.fan_animation_index = (self.fan_animation_index + 1) % len(
                    self.fan_images["on"]
                )
                self.last_fan_toggle_time = time.time()
            fan_image = self.fan_images["on"][self.fan_animation_index]
        screen.blit(fan_image, fan_image.get_rect(topleft=(450, 340)))

    def draw_music_label(self, screen):
        if self.play_music:
            font = pygame.font.Font(None, 36)
            text_surface = font.render("Playing music...", True, WHITE)
            text_rect = text_surface.get_rect(center=(600, 250))
            pygame.draw.rect(screen, BLACK, text_rect.inflate(20, 10))
            screen.blit(text_surface, text_rect)

    def toggle_pause(self):
        self.paused = not self.paused
        self.add_log(f"Live recording {'paused' if self.paused else 'resumed'}.", True)


    def clear_history(self):
        self.log_messages.clear()
        self.add_log("All recorded history cleared.", True)


def parse_label_file(txt_path):
    labels = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        labels.append(parts[2])
                    except ValueError:
                        continue
    except Exception:
        return []
    return labels


def build_label_map(dataset_dir, min_count=10):
    counter = Counter()
    wav_basenames = {
        os.path.splitext(f)[0]
        for f in os.listdir(dataset_dir)
        if f.lower().endswith(".wav")
    }
    for fname in os.listdir(dataset_dir):
        if (
            fname.lower().endswith(".txt")
            and os.path.splitext(fname)[0] in wav_basenames
        ):
            for w in parse_label_file(os.path.join(dataset_dir, fname)):
                counter[w] += 1
    vocab = ["<SILENCE>"] + sorted(
        [word for word, count in counter.items() if count >= min_count]
    )
    return {word: i for i, word in enumerate(vocab)}, {
        i: word for i, word in enumerate(vocab)
    }


def find_latest_checkpoint():
    all_checkpoints = glob.glob("sami_tdnn_step_*.pth") + glob.glob(
        "sami_tdnn_final_step_*.pth"
    )
    if not all_checkpoints:
        return None, 0
    checkpoints_with_step = []
    for fpath in all_checkpoints:
        try:
            step = int(os.path.basename(fpath).split("_")[-1].replace(".pth", ""))
            checkpoints_with_step.append((step, fpath))
        except (ValueError, IndexError):
            continue
    if checkpoints_with_step:
        checkpoints_with_step.sort(key=lambda x: x[0], reverse=True)
        return checkpoints_with_step[0][1], checkpoints_with_step[0][0]
    return None, 0


if not os.path.exists(PANN_PATH):
    sys.exit(f"Error: PANN repository not found at {PANN_PATH}.")
sys.path.append(os.path.abspath(PANN_PATH))
sys.path.append(os.path.abspath(os.path.join(PANN_PATH, "pytorch")))
try:
    spec = importlib.util.spec_from_file_location(
        "models", os.path.join(PANN_PATH, "pytorch/models.py")
    )
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
except Exception as e:
    sys.exit(f"Error loading PANN models: {e}")


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size, self.dilation = kernel_size, dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=0
        )
        self.bn, self.relu, self.se_block = (
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            SEBlock(out_channels),
        )

    def forward(self, x):
        total_padding = self.dilation * (self.kernel_size - 1)
        x = F.pad(x, (total_padding // 2, total_padding - (total_padding // 2)))
        return self.se_block(self.relu(self.bn(self.conv(x))))


class GatedAttentionPooling(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads, batch_first=True
        )
        self.gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())

    def forward(self, x):
        query = torch.mean(x, dim=1, keepdim=True)
        attn_output, _ = self.attn(query, x, x)
        gated = attn_output * self.gate(query)
        return torch.cat(
            (gated.squeeze(1), torch.std(x, dim=1), torch.max(x, dim=1).values), dim=1
        )


class AttentionClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_heads=4, dropout=0.15):
        super().__init__()
        self.projection = nn.Linear(in_dim, hidden_dim)
        self.attention = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        B, D_triple = x.shape
        D = D_triple // 3
        x_reshaped = x.view(B, 3, D)
        projected_x = self.projection(x_reshaped)
        attn_out = self.attention(projected_x)
        pooled_out = torch.mean(attn_out, dim=1)
        return self.fc(self.layer_norm(pooled_out))


class SAMIModel(nn.Module):
    def __init__(
        self,
        n_mels=64,
        num_classes=None,
        config=None,
        PANN_SAMPLE_RATE=32000,
        N_FFT=1024,
    ):
        super(SAMIModel, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be set")
        if config is None:
            config = {
                "tdnn_dim": 512,
                "pooling_heads": 4,
                "classifier_hidden_dim": 256,
                "classifier_heads": 4,
                "dropout": 0.15,
            }
        tdnn_dim, dropout = config["tdnn_dim"], config["dropout"]
        self.pretrained_cnn = models.MobileNetV2(
            sample_rate=PANN_SAMPLE_RATE,
            window_size=N_FFT,
            hop_size=256,
            mel_bins=n_mels,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        self.tdnn_proj = nn.Linear(1280, tdnn_dim)
        self.tdnn_micro = nn.Sequential(
            TDNNLayer(tdnn_dim, tdnn_dim, 1, 1), TDNNLayer(tdnn_dim, tdnn_dim, 1, 2)
        )
        self.tdnn_fast = nn.Sequential(
            TDNNLayer(tdnn_dim, tdnn_dim, 2, 1),
            TDNNLayer(tdnn_dim, tdnn_dim, 2, 2),
            TDNNLayer(tdnn_dim, tdnn_dim, 2, 3),
        )
        self.tdnn_slow = nn.Sequential(
            TDNNLayer(tdnn_dim, tdnn_dim, 3, 1),
            TDNNLayer(tdnn_dim, tdnn_dim, 3, 3),
            TDNNLayer(tdnn_dim, tdnn_dim, 3, 5),
        )
        self.tdnn_fusion = nn.Linear(tdnn_dim * 3, tdnn_dim)
        self.layer_norm = nn.LayerNorm(tdnn_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling = GatedAttentionPooling(
            tdnn_dim, num_heads=config["pooling_heads"]
        )
        self.classifier_head = AttentionClassifierHead(
            in_dim=tdnn_dim,
            hidden_dim=config["classifier_hidden_dim"],
            num_classes=num_classes,
            num_heads=config["classifier_heads"],
            dropout=dropout,
        )

    def forward(self, waveform):
        x = self.pretrained_cnn.logmel_extractor(
            self.pretrained_cnn.spectrogram_extractor(waveform)
        )
        x = self.pretrained_cnn.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = self.pretrained_cnn.features(x)
        features = torch.mean(x, dim=3).transpose(1, 2)
        tdnn_input = self.tdnn_proj(features)
        tdnn_permuted = tdnn_input.permute(0, 2, 1)
        micro_out, fast_out, slow_out = (
            self.tdnn_micro(tdnn_permuted),
            self.tdnn_fast(tdnn_permuted),
            self.tdnn_slow(tdnn_permuted),
        )
        fused_out = self.tdnn_fusion(
            torch.cat((micro_out, fast_out, slow_out), dim=1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        tdnn_out = self.layer_norm(fused_out.permute(0, 2, 1) + tdnn_input)
        return self.classifier_head(self.pooling(self.dropout(tdnn_out)))


class AudioRecorder(threading.Thread):
    def __init__(
        self, audio_queue, target_sr, chunk_size, format, channels, game_state
    ):
        super().__init__()
        (
            self.audio_queue,
            self.target_sr,
            self.chunk_size,
            self.format,
            self.channels,
        ) = (audio_queue, target_sr, chunk_size, format, channels)
        self.p, self._stop_event, self.game_state = (
            pyaudio.PyAudio(),
            threading.Event(),
            game_state,
        )
        default_idx = -1
        self.game_state.audio_queue_ref = audio_queue
        try:
            default_idx = self.p.get_default_input_device_info()["index"]
        except IOError:
            game_state.add_log("WARNING: No default input device found.", True)

    def run(self):
        self.game_state.add_log("Audio recording starting...", True)
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=DEFAULT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            self.game_state.add_log("🎤 Microphone active. Speak now!", True)
            while not self._stop_event.is_set():
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    if self.game_state.system_busy:
                        continue
                    self.audio_queue.put(data)
                except Exception as e:
                    self.game_state.add_log(f"Audio read error: {e}", True)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            self.game_state.add_log(f"CRITICAL: Mic init failed: {e}", True)
        finally:
            self.p.terminate()
            self.game_state.add_log("Audio recording stopped.", True)

    def stop(self):
        self._stop_event.set()


def check_sequence(words: list, sequence: list, max_gap: int) -> bool:
    if not words or not sequence or len(words) < len(sequence):
        return False
    words_upper, seq_upper = [w.upper() for w in words], [s.upper() for s in sequence]
    for i in range(len(words_upper) - len(seq_upper) + 1):
        if words_upper[i] == seq_upper[0]:
            last_word_index, is_match = i, True
            for k in range(1, len(seq_upper)):
                target_word, found_in_window = seq_upper[k], False
                for j in range(
                    last_word_index + 1,
                    min(last_word_index + max_gap + 2, len(words_upper)),
                ):
                    if words_upper[j] == target_word:
                        last_word_index, found_in_window = j, True
                        break
                if not found_in_window:
                    is_match = False
                    break
            if is_match:
                return True
    return False


class InferenceProcessor(threading.Thread):
    def __init__(
        self,
        model,
        audio_queue,
        label_map_rev,
        max_window_seconds,
        hop_samples,
        target_sr,
        device,
        game_state,
        recorder,
        weather_cache,
        log_queue,
    ):
        super().__init__()
        self.model, self.audio_queue, self.label_map_rev = (
            model,
            audio_queue,
            label_map_rev,
        )
        self.buffer_max_samples = int(max_window_seconds * target_sr)
        self.hop_samples, self.target_sr, self.device = hop_samples, target_sr, device
        self._stop_event, self.audio_buffer = threading.Event(), np.zeros(
            0, dtype=np.int16
        )
        self.new_samples_counter, self.game_state, self.recorder = (
            0,
            game_state,
            recorder,
        )
        self.base_threshold = 0.80
        self.game_state.set_predictions([], self.base_threshold, 0.0)
        self.word_history, self.weather_cache, self.log_queue = (
            deque(maxlen=10),
            weather_cache,
            log_queue,
        )
        self.weak_detection_buffer = defaultdict(
            lambda: deque(maxlen=WEAK_DETECTION_WINDOW)
        )

        self.pending_command = None  # "CALL" or "TIMER"
        self.pending_expiry = (
            0  # How many subsequent words to listen to before timing out
        )

    def set_context(self, command_type, response_sound=None):
        if response_sound and os.path.exists(response_sound):
            threading.Thread(
                target=self.game_state._play_audio_blocking, args=(response_sound,)
            ).start()

        self.pending_command = command_type
        self.pending_expiry = 5  # Look at the next 5 confirmed/promoted words
        self.game_state.add_log(f"Listening for {command_type} details...", True)
        # Clear history so the trigger words don't confuse the next logic
        self.word_history.clear()

    def process_commands(self):
        words = list(self.word_history)
        if not words:
            return False

        last_word = words[-1]
        last_word_upper = last_word.upper()

        if self.pending_command:
            self.pending_expiry -= 1

            if self.pending_command == "CALL":
                target_names = ["ANIL", "MIRA", "SUMINA"]
                if last_word_upper in target_names:
                    self.game_state.call_person(last_word.title())
                    self.pending_command = None
                    self.word_history.clear()
                    return True

            elif self.pending_command == "TIMER":
                valid_numbers = [str(i) for i in range(1, 11)]

                if last_word_upper in valid_numbers:
                    try:
                        seconds = int(last_word_upper)
                        self.game_state.create_timer(seconds)
                        self.pending_command = None
                        self.word_history.clear()
                        return True
                    except ValueError:
                        pass

            if self.pending_expiry <= 0:
                self.game_state.add_log(
                    f"Command '{self.pending_command}' timed out.", True
                )
                self.pending_command = None

        command_map = [
            (lambda: self.set_context("CALL"), [["PHONE", "GARA"], ["CALL", "GARA"]]),
            (
                lambda: self.set_context(
                    "TIMER",
                    os.path.join(
                        "assets", "Responses", "kati_minute_ko_timer_rakhdim.wav"
                    ),
                ),
                [["TIMER", "RAKHA"], ["TIMER", "CHALAU"], ["TIMER", "ON"]],
            ),
            (
                lambda: self.game_state.toggle_light(),
                [
                    ["BATTI", "BALA"],
                    ["BATTI", "CHALAU"],
                    ["BATTI", "ON", "GARA"],
                    ["LIGHT", "ON"],
                ],
            ),
            (
                lambda: self.game_state.toggle_light(),
                [["BATTI", "NIBHAU"], ["LIGHT", "OFF"]],
            ),
            (
                lambda: self.game_state.set_door_state("open"),
                [["DHOKA", "KHOLA"], ["DHOKA", "ON", "GARA"]],
            ),
            (
                lambda: self.game_state.set_door_state("closed"),
                [["DHOKA", "BANDA", "GARA"]],
            ),
            (
                lambda: self.game_state.set_tv_state("on"),
                [["TV", "ON"], ["TV", "KHOLA"], ["TV", "CHALAU"]],
            ),
            (
                lambda: self.game_state.set_tv_state("off"),
                [["TV", "OFF"], ["TV", "BANDA"], ["TV", "NIBHAU"]],
            ),
            (
                lambda: self.game_state.toggle_time_display(),
                [["KATI", "BAJYO"], ["TIME", "DEKHAU"]],
            ),
            (
                lambda: self.game_state.set_weather_info(self.weather_cache),
                [["WEATHER", "KASTO", "6"], ["MAUSAM", "KASTO", "6"], ["MAUSAM", "6"]],
            ),
            (
                lambda: self.game_state.toggle_fan(True),
                [["FAN", "CHALAU"], ["FAN", "ON", "GARA"], ["PANKHA", "ON"]],
            ),
            (
                lambda: self.game_state.toggle_fan(False),
                [["FAN", "NIBHAU"], ["PANKHA", "BANDA"]],
            ),
            (lambda: self.game_state.create_alarm(), [["ALARM", "RAKHA"]]),
            (
                lambda: self.game_state.toggle_music(True),
                [["GEET", "BAJAU"], ["GEET", "CHALAU"], ["GEET", 'CHALAUXA']],
            ),
            (
                lambda: self.game_state.toggle_music(False),
                [["GEET", "BANDA"], ["GEET", "NIBHAU"]],
            ),
            (
                lambda: (
                    self.game_state.show_toast("Opening Facebook...", 5),
                    self.game_state.add_log("Opening Facebook...", True),
                ),
                [["FACEBOOK", "CHALAU"], ["FACEBOOK", "KHOLA"], ["FACEBOOK", "ON"]],
            ),
        ]

        for action, phrases in command_map:
            for phrase in phrases:
                if check_sequence(words, phrase, max_gap=3):
                    action()
                    if self.pending_command is None:
                        self.word_history.clear()
                    self.game_state.add_log("--> Command detected!", True)
                    return True
        return False

    def process_buffer(self):
        if (
            self.game_state.paused
            or self.game_state.system_busy
            or len(self.audio_buffer) < self.buffer_max_samples
        ):
            return

        with torch.no_grad():
            waveforms = [
                torch.from_numpy(
                    self.audio_buffer[-int(d * self.target_sr) :].astype(np.float32)
                    / 32768.0
                )
                for d in INFERENCE_WINDOW_DURATIONS_S
            ]
            resampled = [
                torchaudio.functional.resample(w, self.target_sr, PANN_SAMPLE_RATE)
                for w in waveforms
            ]
            batch = torch.stack(
                [F.pad(r, (0, WINDOW_SAMPLES_MAX - r.shape[0])) for r in resampled]
            ).to(self.device)
            avg_probs = torch.sigmoid(self.model(batch)).mean(dim=0)

        confident_predictions = sorted(
            [
                (self.label_map_rev.get(i), p.item())
                for i, p in enumerate(avg_probs)
                if p > self.base_threshold and i != 0
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        self.game_state.set_predictions(
            confident_predictions, self.base_threshold, avg_probs[0].item()
        )

        strong_words_this_frame = set()
        if confident_predictions:
            for word, prob in confident_predictions:
                strong_words_this_frame.add(word)
                if not self.word_history or self.word_history[-1] != word:
                    self.word_history.append(word)
                if word in self.weak_detection_buffer:
                    del self.weak_detection_buffer[word]
            self.log_queue.put(f"Word History Queue: {list(self.word_history)}")
            if self.process_commands():
                return

        for i, p in enumerate(avg_probs[1:], 1):
            prob = p.item()
            word = self.label_map_rev.get(i)
            if not word:
                continue

            trigger_thresh = WEAK_DETECTION_THRESHOLD
            if word in MANUAL_THRESHOLDS:
                trigger_thresh = MANUAL_THRESHOLDS[word][0]

            if prob > trigger_thresh:
                if word not in strong_words_this_frame:
                    self.weak_detection_buffer[word].append(prob)

        promoted_words = False
        for word in list(self.weak_detection_buffer.keys()):
            prob_deque = self.weak_detection_buffer[word]
            if len(prob_deque) >= WEAK_DETECTION_FRAMES:
                avg_prob = sum(prob_deque) / len(prob_deque)

                promo_thresh = WEAK_DETECTION_AVG_PROB
                if word in MANUAL_THRESHOLDS:
                    promo_thresh = MANUAL_THRESHOLDS[word][1]

                if avg_prob >= promo_thresh:
                    self.log_queue.put(
                        f"Promoted weak detection: '{word}' (avg: {avg_prob:.2f} >= {promo_thresh})"
                    )
                    if not self.word_history or self.word_history[-1] != word:
                        self.word_history.append(word)
                        promoted_words = True
                    del self.weak_detection_buffer[word]

        if promoted_words:
            self.log_queue.put(f"Word History Queue: {list(self.word_history)}")
            self.process_commands()

    def run(self):
        self.log_queue.put("--- Starting Inference Processor ---")
        while not self._stop_event.is_set():
            try:
                new_samples = np.frombuffer(
                    self.audio_queue.get(timeout=0.05), dtype=np.int16
                )
                self.audio_buffer = np.concatenate([self.audio_buffer, new_samples])[
                    -self.buffer_max_samples :
                ]
                self.new_samples_counter += len(new_samples)
                if self.new_samples_counter >= self.hop_samples:
                    self.process_buffer()
                    self.new_samples_counter = 0
            except queue.Empty:
                if self.recorder._stop_event.is_set() and self.audio_queue.empty():
                    break
        self.log_queue.put("--- Inference Processor Stopped ---")

    def stop(self):
        self._stop_event.set()


def load_image(path, size=None, convert_with_alpha=False):
    image = pygame.image.load(path)
    image = image.convert_alpha() if convert_with_alpha else image.convert()
    if size:
        image = pygame.transform.smoothscale(image, size)
    return image


def draw_info_panel(screen, font, game_state):
    panel = pygame.Surface((INFO_PANEL_WIDTH, WINDOW_HEIGHT))
    panel.fill(GRAY)
    predictions, threshold, silence_prob = game_state.get_predictions()
    y = 10
    panel.blit(font.render("Latest Predictions", True, BLACK), (10, y))
    y += 30
    if predictions:
        for label, prob in predictions[:5]:
            panel.blit(font.render(f"{label}: {prob:.2f}", True, BLACK), (10, y))
            y += 28
    else:
        panel.blit(font.render("Listening...", True, BLACK), (10, y))
        y += 28

    status = "Busy (Speaking/Video)" if game_state.system_busy else "Active (Listening)"
    color = (200, 0, 0) if game_state.system_busy else (0, 150, 0)
    panel.blit(font.render(f"Status: {status}", True, color), (10, y))
    y += 28

    panel.blit(
        font.render(
            f"Threshold: {threshold:.2f} | Silence: {silence_prob:.2f}", True, BLACK
        ),
        (10, y),
    )
    y += 40
    panel.blit(font.render("Activity Log", True, BLACK), (10, y))
    y += 30
    for msg in game_state.get_logs(12):
        panel.blit(font.render(msg, True, BLACK), (10, y))
        y += 24
    screen.blit(panel, (GAME_UI_WIDTH, 0))


def run_game_loop(game_state, recorder, processor, log_queue):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("SAMI")
    clock, font, big_font = (
        pygame.time.Clock(),
        pygame.font.SysFont("Arial", 24),
        pygame.font.SysFont("Arial", 48, bold=True),
    )
    try:
        assets = {
            "house_closed": load_image(
                os.path.join("assets", "DefaultHouse.jpg"),
                (GAME_UI_WIDTH, WINDOW_HEIGHT),
            ),
            "house_open": load_image(
                os.path.join("assets", "DefaultHouseDoorOpen.jpg"),
                (GAME_UI_WIDTH, WINDOW_HEIGHT),
            ),
            "tv_default": load_image(
                os.path.join("assets", "Tv.png"), (TV_RECT_W, TV_RECT_H), True
            ),
            "tv_on": load_image(
                os.path.join("assets", "TvOn.png"), (TV_RECT_W, TV_RECT_H), True
            ),
            "speaker": load_image(
                os.path.join("assets", "speaker.png"), (140, 140), True
            ),
        }
    except (pygame.error, FileNotFoundError) as e:
        game_state.add_log(f"ERROR: Failed to load assets: {e}", True)
        pygame.quit()
        return
    light_off_overlay = pygame.Surface((GAME_UI_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    light_off_overlay.fill((0, 0, 0, 150))
    game_state.add_log("UI ready.", True)
    running = True
    while running:
        while not log_queue.empty():
            try:
                print(log_queue.get_nowait())
            except queue.Empty:
                break

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    game_state.toggle_pause()
                elif event.key == pygame.K_c:
                    game_state.clear_history()

        d_state, l_state, t_state, s_time, s_weather, w_desc = game_state.get_states()
        screen.blit(
            assets["house_open"] if d_state == "open" else assets["house_closed"],
            (0, 0),
        )
        screen.blit(
            assets["tv_on"] if t_state == "on" else assets["tv_default"],
            (TV_RECT_X, TV_RECT_Y),
        )
        screen.blit(assets["speaker"], (600, 340))
        game_state.draw_fan(screen)
        game_state.draw_music_label(screen)
        if l_state == "off":
            screen.blit(light_off_overlay, (0, 0))
        if s_time or s_weather:
            info_y = 50
            if s_time:
                time_surf = big_font.render(time.strftime("%H:%M:%S"), True, WHITE)
                time_rect = time_surf.get_rect(center=(GAME_UI_WIDTH // 2, info_y))
                pygame.draw.rect(screen, BLUE, time_rect.inflate(20, 10))
                screen.blit(time_surf, time_rect)
                info_y += 70
            if s_weather:
                weather_surf = big_font.render(w_desc, True, WHITE)
                weather_rect = weather_surf.get_rect(
                    center=(GAME_UI_WIDTH // 2, info_y)
                )
                pygame.draw.rect(screen, BLUE, weather_rect.inflate(20, 10))
                screen.blit(weather_surf, weather_rect)

        draw_info_panel(screen, font, game_state)
        game_state.draw_toast(screen)
        pygame.display.flip()
        clock.tick(30)
        if not recorder.is_alive() or not processor.is_alive():
            game_state.add_log("Audio threads stopped. Closing UI.", True)
            running = False
    pygame.quit()


def model_summary(model, log_queue):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_queue.put(
        f"\nModel Summary: {model.__class__.__name__}\n"
        f"  - Total params:     {total:,}\n"
        f"  - Trainable params: {trainable:,}"
    )


if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        sys.exit(f"ERROR: Dataset directory not found at {DATASET_DIR}.")

    log_queue = queue.Queue()
    game_state = GameState(log_queue=log_queue)

    print("Building label map...")
    label_map, label_map_rev = build_label_map(DATASET_DIR, min_count=10)

    sami_model = SAMIModel(
        n_mels=N_MELS, num_classes=len(label_map), config=MODEL_CONFIG_LIGHTWEIGHT
    ).to(DEVICE)
    sami_model.eval()

    model_summary(sami_model, log_queue)

    ckpt_path, step = find_latest_checkpoint()
    if ckpt_path:
        log_queue.put(f"Loading checkpoint: {ckpt_path} (step {step})")
        try:
            sami_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            log_queue.put(f"✓ Model loaded successfully.")
        except Exception as e:
            sys.exit(f"✗ ERROR loading checkpoint: {e}\nArchitecture mismatch likely.")
    else:
        sys.exit("✗ ERROR: No checkpoint file found.")

    weather_description_cache = get_weather_in_kathmandu_description(log_queue)

    audio_q = queue.Queue()
    recorder = AudioRecorder(
        audio_q, DEFAULT_SAMPLE_RATE, CHUNK_SIZE, FORMAT, CHANNELS, game_state
    )
    processor = InferenceProcessor(
        sami_model,
        audio_q,
        label_map_rev,
        MAX_WINDOW_SECONDS,
        HOP_SAMPLES,
        DEFAULT_SAMPLE_RATE,
        DEVICE,
        game_state,
        recorder,
        weather_cache=weather_description_cache,
        log_queue=log_queue,
    )

    recorder.start()
    processor.start()

    try:
        run_game_loop(game_state, recorder, processor, log_queue)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recorder.stop()
        processor.stop()
        recorder.join()
        processor.join()
        print("Inference stopped. Goodbye!")
