"""
疲労検出アプリ
"""
import sys
import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import os
import time
import cv2
import csv
from datetime import datetime, date, timedelta
import subprocess
import tempfile
from threading import Thread
import requests
import base64
from collections import deque
import pandas as pd

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QMenu, QSystemTrayIcon, QDialog, QCheckBox,
                             QPushButton, QDialogButtonBox, QLabel, QHBoxLayout, QLineEdit,
                             QSpinBox, QFormLayout, QFrame)
from PyQt6.QtCore import QThread, QObject, pyqtSignal, QTimer, Qt, QSettings
from PyQt6.QtGui import QIcon, QAction, QFont

import pyqtgraph as pg
import subprocess

from models_detection import RetinaNet, predict_post_process

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

# ================================================================
# アプリケーション設定
# ================================================================
DETECTION_MODEL_PATH = resource_path(os.path.join('resources', 'trained_parameters', 'YawnAndEye.pth'))
DETECTION_NUM_CLASSES = 5
DETECTION_CLASS_NAMES = ["closed_eye", "closed_mouth", "open_eye", "open_mouth", "wake-drowsy"]
OPEN_JTALK_DIC_PATH = resource_path(os.path.join('resources', 'dic'))
OPEN_JTALK_VOICE_PATH = resource_path(os.path.join('resources', 'voice', 'mei', 'mei_normal.htsvoice'))
CAPTURE_INTERVAL_SEC = 2
APP_NAME = "AMaGaMi"
APP_SUPPORT_DIR = os.path.expanduser(os.path.join("~", "Library", "Application Support", APP_NAME))
os.makedirs(APP_SUPPORT_DIR, exist_ok=True)
LOG_FILE = os.path.join(APP_SUPPORT_DIR, 'fatigue_log.csv')

# ================================================================
# Toggl連携マネージャー
# ================================================================
class TogglManager:
    BASE_URL = "https://api.track.toggl.com/api/v9"
    def __init__(self):
        self.api_token = None
        self.workspace_id = None
        self.current_entry_id = None
        self.is_enabled = False

    def configure(self, api_token, is_enabled):
        self.api_token = api_token
        self.is_enabled = is_enabled
        if self.is_enabled and self.api_token:
            self.workspace_id = self._get_workspace_id()

    def _get_headers(self):
        if not self.api_token:
            return None
        token_b64 = base64.b64encode(f"{self.api_token}:api_token".encode()).decode()
        return {"Content-Type": "application/json", "Authorization": f"Basic {token_b64}"}

    def _get_workspace_id(self):
        try:
            response = requests.get(f"{self.BASE_URL}/me/workspaces", headers=self._get_headers())
            response.raise_for_status()
            workspaces = response.json()
            if workspaces:
                print(f"Toggl: ワークスペース「{workspaces[0]['name']}」を使用します。")
                return workspaces[0]['id']
            else:
                print("Togglエラー: 利用可能なワークスペースが見つかりません。")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Togglエラー: ワークスペースの取得に失敗しました: {e}")
            return None

    def start_session(self, description="疲労検出セッション"):
        if not self.is_enabled or not self.workspace_id:
            return
        data = {"description": description, "workspace_id": self.workspace_id, "start": datetime.utcnow().isoformat() + "Z", "duration": -1, "created_with": "FatigueMonitorApp"}
        try:
            response = requests.post(f"{self.BASE_URL}/workspaces/{self.workspace_id}/time_entries", headers=self._get_headers(), json=data)
            response.raise_for_status()
            entry_data = response.json()
            self.current_entry_id = entry_data['id']
            print(f"Toggl: 時間記録を開始しました (ID: {self.current_entry_id})")
        except requests.exceptions.RequestException as e:
            print(f"Togglエラー: 時間記録の開始に失敗しました: {e}")

    def stop_session(self):
        if not self.is_enabled or not self.current_entry_id:
            return
        try:
            response = requests.patch(f"{self.BASE_URL}/workspaces/{self.workspace_id}/time_entries/{self.current_entry_id}/stop", headers=self._get_headers())
            response.raise_for_status()
            print(f"Toggl: 時間記録を停止しました (ID: {self.current_entry_id})")
            self.current_entry_id = None
        except requests.exceptions.RequestException as e:
            print(f"Togglエラー: 時間記録の停止に失敗しました: {e}")

# ================================================================
# 各種マネージャークラス
# ================================================================
class FatigueScorer:
    def __init__(self, settings):
        self.settings = settings
        self.score = 0
        self.score_cap = 300

    def update_score(self, analysis_results):
        self.score = max(0, self.score - self.settings['decay_rate'])
        if 'open_mouth' in analysis_results['detections']:
            self.score += self.settings['yawn_score']
        if analysis_results['is_microsleep']:
            self.score += self.settings['microsleep_score']
        if analysis_results['perclos'] > 0.3:
            self.score += self.settings['perclos_score']
        self.score = min(self.score, self.score_cap)
        return self.score

    def reset(self):
        self.score = 0

class Notifier:
    def __init__(self):
        self.use_macos_notification = True
        self.use_voice_notification = True

    def issue_warning(self, score, reason="fatigue"):
        if reason == "fatigue":
            message = f"疲労度が {int(score)} になりました。少し休憩しませんか？"
            voice_message = "疲労が蓄積しているようです。少し休憩しませんか。"
        elif reason == "duration":
            message = "90分以上連続で作業しています。そろそろ休憩しませんか？"
            voice_message = "90分以上連続で作業しています。そろそろ休憩しませんか。"
        elif reason == "sustained_fatigue":
            message = "高い疲労状態が5分以上続いています。作業を中断し、休憩しましょう。"
            voice_message = "かなりお疲れのようです。作業を中断して、休憩してください。"
        else:
            return

        print(f"★★★ 警告: {message} ★★★")
        if self.use_macos_notification:
            try:
                title = "疲労警告!!"
                script = f'display notification "{message}" with title "{title}"'
                subprocess.run(["osascript", "-e", script])
            except Exception as e:
                print(f"AppleScriptでの通知エラー: {e}")
        
        if self.use_voice_notification:
            Thread(target=self._speak, args=(voice_message,)).start()

    def _speak(self, text):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                temp_wav_path = fp.name

            open_jtalk_executable = resource_path(os.path.join('resources', 'open_jtalk'))

            os.chmod(open_jtalk_executable, 0o755)
            
            command = [
                open_jtalk_executable,
                '-x', OPEN_JTALK_DIC_PATH,
                '-m', OPEN_JTALK_VOICE_PATH,
                '-ow', temp_wav_path
            ]
            
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            process.communicate(input=text.encode('utf-8'))
            process.wait()
            
            if os.path.exists(temp_wav_path):
                subprocess.run(['/usr/bin/afplay', temp_wav_path])
                os.remove(temp_wav_path)
        except Exception as e:
            print(f"音声合成エラー: {e}")

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(dict)

    def __init__(self, model, transform, device):
        super().__init__()
        self.detection_model = model
        self.detection_transform = transform
        self.device = device
        self.is_running = True
        self.perclos_window = deque(maxlen=int(60 / CAPTURE_INTERVAL_SEC))

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("エラー: カメラを起動できませんでした。")
            self.finished.emit()
            return
        
        while self.is_running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue
            
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = self._predict_detection(image_pil)
            
            is_eye_closed = 'closed_eye' in detections

            self.perclos_window.append(1 if is_eye_closed else 0)
            perclos = sum(self.perclos_window) / len(self.perclos_window) if self.perclos_window else 0.0

            is_microsleep = perclos > 0.8

            analysis_results = {
                "detections": detections,
                "is_microsleep": is_microsleep,
                "perclos": perclos
            }
            self.progress.emit(analysis_results)
            
            elapsed_time = time.time() - start_time
            sleep_time = max(0, CAPTURE_INTERVAL_SEC - elapsed_time)
            time.sleep(sleep_time)
            
        cap.release()
        self.finished.emit()

    def _predict_detection(self, image):
        img_tensor = self.detection_transform(image).unsqueeze(0).to(self.device)
        _, _, h, w = img_tensor.shape
        target_h, target_w = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
        img_tensor = F.pad(img_tensor, (0, target_w - w, 0, target_h - h))
        with torch.no_grad():
            preds_class, preds_box, anchors = self.detection_model(img_tensor)

        _, _, final_labels = predict_post_process(
            preds_class[0], preds_box[0], anchors, conf_threshold=0.5, nms_threshold=0.5
        )
        return [DETECTION_CLASS_NAMES[label.item()] for label in final_labels]

# ================================================================
# UIウィンドウクラス
# ================================================================
class GraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("疲労度グラフ")
        self.setGeometry(200, 200, 800, 400)
        self.graph_widget = pg.PlotWidget()
        self.setCentralWidget(self.graph_widget)
        self.graph_widget.setBackground('w')
        self.graph_widget.setLabel('left', '疲労スコア', color='black')
        self.graph_widget.setLabel('bottom', '時刻', color='black')
        self.graph_widget.showGrid(x=True, y=True)
        self.graph_widget.setAxisItems({'bottom': pg.DateAxisItem()})
        self.data_line = self.graph_widget.plot([], [], pen=pg.mkPen(color=(255, 0, 0), width=3))
        self.timestamps, self.scores = [], []

    def update_graph(self, timestamp, score):
        self.timestamps.append(timestamp)
        self.scores.append(score)
        max_points = int(3600 / CAPTURE_INTERVAL_SEC)
        if len(self.timestamps) > max_points:
            self.timestamps.pop(0)
            self.scores.pop(0)
        self.data_line.setData(self.timestamps, self.scores)

    def clear_graph(self):
        self.timestamps, self.scores = [], []
        self.data_line.setData([], [])

class DashboardWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("疲労度分析")
        self.setGeometry(300, 300, 400, 300)
        
        layout = QVBoxLayout(self)
        self.stats_label = QLabel("統計情報を読み込んでいます…")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.stats_label)
        
        refresh_button = QPushButton("更新")
        refresh_button.clicked.connect(self.refresh_stats)
        layout.addWidget(refresh_button)

    def refresh_stats(self):
        if not os.path.exists(LOG_FILE):
            self.stats_label.setText("ログファイルが見つかりません。")
            return
        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty:
                self.stats_label.setText("ログデータがありません。")
                return
            
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            today_df = df[df['Timestamp'].dt.date == date.today()]
            if not today_df.empty:
                peak_fatigue_today = today_df.loc[today_df['FatigueScore'].idxmax()]
                peak_time_str = peak_fatigue_today['Timestamp'].strftime('%H:%M')
                peak_score = int(peak_fatigue_today['FatigueScore'])
                today_stats = f"今日最も疲労度が高かったのは<b>{peak_time_str}</b>頃です (疲労度: {peak_score})。"
            else:
                today_stats = "今日の作業ログはまだありません。"

            start_of_week = date.today() - timedelta(days=date.today().weekday())
            week_df = df[df['Timestamp'].dt.date >= start_of_week]
            if not week_df.empty:
                avg_score_week = int(week_df['FatigueScore'].mean())
                week_stats = f"今週の平均疲労度は<b>{avg_score_week}</b>です。"
            else:
                week_stats = "今週の作業ログはまだありません。"

            self.stats_label.setText(f"<h2>疲労傾向分析</h2><p>{today_stats}</p><p>{week_stats}</p>")

        except Exception as e:
            self.stats_label.setText(f"統計情報の読み込み中にエラーが発生しました:\n{e}")

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_stats()

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.settings = QSettings("MyCompany", "FatigueMonitor")
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.yawn_score_spin = QSpinBox(); self.yawn_score_spin.setRange(0, 200); form_layout.addRow("あくびの値:", self.yawn_score_spin)
        self.perclos_score_spin = QSpinBox(); self.perclos_score_spin.setRange(0, 200); form_layout.addRow("PERCLOSスコア:", self.perclos_score_spin)
        self.microsleep_score_spin = QSpinBox(); self.microsleep_score_spin.setRange(0, 200); form_layout.addRow("居眠りの値:", self.microsleep_score_spin)
        self.decay_rate_spin = QSpinBox(); self.decay_rate_spin.setRange(0, 200); form_layout.addRow("スコア自然減衰量:", self.decay_rate_spin)
        self.fatigue_threshold_spin = QSpinBox(); self.fatigue_threshold_spin.setRange(0, 200); form_layout.addRow("警告閾値:", self.fatigue_threshold_spin)
        
        layout.addWidget(QLabel("<b>スコアリング設定</b>"))
        layout.addLayout(form_layout)
        layout.addWidget(self.create_separator())

        self.cb_macos_notify = QCheckBox("Mac上の通知を有効にする")
        self.cb_voice_notify = QCheckBox("声による通知を有効にする")
        layout.addWidget(QLabel("<b>通知設定</b>"))
        layout.addWidget(self.cb_macos_notify)
        layout.addWidget(self.cb_voice_notify)
        layout.addWidget(self.create_separator())

        layout.addWidget(QLabel("<b>Toggl Track 連携</b>"))
        self.cb_toggl = QCheckBox("Toggl Trackでの記録を有効にする")
        layout.addWidget(self.cb_toggl)
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("API Token:"))
        self.le_toggl_token = QLineEdit(); self.le_toggl_token.setEchoMode(QLineEdit.EchoMode.Password)
        api_layout.addWidget(self.le_toggl_token)
        layout.addLayout(api_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.load_settings()

    def create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def load_settings(self):
        self.yawn_score_spin.setValue(self.settings.value("yawn_score", 50, type=int))
        self.perclos_score_spin.setValue(self.settings.value("perclos_score", 5, type=int))
        self.microsleep_score_spin.setValue(self.settings.value("microsleep_score", 100, type=int))
        self.decay_rate_spin.setValue(self.settings.value("decay_rate", 5, type=int))
        self.fatigue_threshold_spin.setValue(self.settings.value("fatigue_threshold", 100, type=int))
        self.cb_macos_notify.setChecked(self.settings.value("use_macos_notification", True, type=bool))
        self.cb_voice_notify.setChecked(self.settings.value("use_voice_notification", True, type=bool))
        self.cb_toggl.setChecked(self.settings.value("use_toggl", False, type=bool))
        self.le_toggl_token.setText(self.settings.value("toggl_api_token", "", type=str))

    def accept(self):
        self.settings.setValue("yawn_score", self.yawn_score_spin.value())
        self.settings.setValue("perclos_score", self.perclos_score_spin.value())
        self.settings.setValue("microsleep_score", self.microsleep_score_spin.value())
        self.settings.setValue("decay_rate", self.decay_rate_spin.value())
        self.settings.setValue("fatigue_threshold", self.fatigue_threshold_spin.value())
        self.settings.setValue("use_macos_notification", self.cb_macos_notify.isChecked())
        self.settings.setValue("use_voice_notification", self.cb_voice_notify.isChecked())
        self.settings.setValue("use_toggl", self.cb_toggl.isChecked())
        self.settings.setValue("toggl_api_token", self.le_toggl_token.text())
        super().accept()

class StopwatchWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ストップウォッチ")
        self.setFixedSize(300, 150)
        layout = QVBoxLayout(self)
        self.time_label = QLabel("00:00:00")
        font = QFont("Menlo", 40, QFont.Weight.Bold); self.time_label.setFont(font)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(self.time_label)
        self.timer = QTimer(self); self.timer.timeout.connect(self.update_time); self.elapsed_seconds = 0

    def start(self):
        self.elapsed_seconds = 0
        self.update_time()
        self.timer.start(1000)

    def stop(self):
        self.timer.stop()

    def reset(self):
        self.stop()
        self.elapsed_seconds = 0
        self.time_label.setText("00:00:00")

    def update_time(self):
        self.elapsed_seconds += 1
        hours = self.elapsed_seconds // 3600
        minutes = (self.elapsed_seconds % 3600) // 60
        seconds = self.elapsed_seconds % 60
        self.time_label.setText(f"{hours:02}:{minutes:02}:{seconds:02}")

# ================================================================
# メインアプリケーションクラス
# ================================================================
class FatigueMonitorApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.detection_model = None
        self.detection_transform = None
        self.thread, self.worker = None, None
        self.is_monitoring = False
        
        self.warning_issued = False
        self.high_score_start_time = None
        self.break_warning_issued = False
        self.sustained_warning_issued = False

        icon_path = resource_path(os.path.join("resources", "icon", "app_icon.icns"))
        icon_path_fatigue = resource_path(os.path.join("resources", "icon", "app_icon_fatigue.icns"))
        self.icon_normal = QIcon(icon_path)
        self.icon_fatigue = QIcon(icon_path_fatigue)
        self.current_icon_state = 'normal'

        self.scorer = None
        self.notifier = Notifier()
        self.toggl_manager = TogglManager()

        self.load_settings_and_init()
        self.load_models()
        self.setup_transforms()
        
        self.graph_window = GraphWindow()
        self.settings_window = SettingsWindow()
        self.dashboard_window = DashboardWindow()
        self.stopwatch_window = StopwatchWindow()
        self.tray_icon = QSystemTrayIcon(self)
        
        self.tray_icon.setIcon(self.icon_normal)
        self.tray_icon.setVisible(True)
        self.tray_icon.setToolTip("疲労検出システム 停止中")
        
        menu = QMenu()
        self.toggle_action = QAction("開始", self, triggered=self.toggle_monitoring)
        menu.addAction(self.toggle_action)
        menu.addAction(QAction("疲労度グラフを表示", self, triggered=self.graph_window.show))
        menu.addAction(QAction("疲労度分析を表示", self, triggered=self.dashboard_window.show))
        menu.addAction(QAction("ストップウォッチを表示", self, triggered=self.stopwatch_window.show))
        menu.addAction(QAction("設定", self, triggered=self.show_settings))
        menu.addSeparator()
        menu.addAction(QAction("終了", self, triggered=self.quit))
        self.tray_icon.setContextMenu(menu)

    def load_settings_and_init(self):
        settings = QSettings("MyCompany", "FatigueMonitor")
        self.notifier.use_macos_notification = settings.value("use_macos_notification", True, type=bool)
        self.notifier.use_voice_notification = settings.value("use_voice_notification", True, type=bool)
        
        use_toggl = settings.value("use_toggl", False, type=bool)
        api_token = settings.value("toggl_api_token", "", type=str)
        self.toggl_manager.configure(api_token, use_toggl)

        scoring_settings = {
            'yawn_score': settings.value("yawn_score", 50, type=int),
            'perclos_score': settings.value("perclos_score", 5, type=int),
            'microsleep_score': settings.value("microsleep_score", 100, type=int),
            'decay_rate': settings.value("decay_rate", 5, type=int),
            'fatigue_threshold': settings.value("fatigue_threshold", 100, type=int)
        }
        self.scorer = FatigueScorer(scoring_settings)

    def load_models(self):
        try:
            self.detection_model = RetinaNet(num_classes=DETECTION_NUM_CLASSES)
            self.detection_model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location=self.device))
            self.detection_model.to(self.device).eval()
        except FileNotFoundError:
            print(f"エラー: モデルファイルが見つかりません: {DETECTION_MODEL_PATH}")
            sys.exit(1)

    def setup_transforms(self):
        detection_mean, detection_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.detection_transform = T.Compose([T.ToTensor(), T.Normalize(mean=detection_mean, std=detection_std)])

    def toggle_monitoring(self):
        if not self.is_monitoring:
            self.start_monitoring()
        else:
            self.stop_monitoring()
            
    def start_monitoring(self):
        self.is_monitoring = True
        self.scorer.reset()
        self.graph_window.clear_graph()
        self.stopwatch_window.start()
        self.toggl_manager.start_session()
        self.warning_issued = False
        self.high_score_start_time = None
        self.break_warning_issued = False
        self.sustained_warning_issued = False

        if self.current_icon_state != 'normal':
            self.tray_icon.setIcon(self.icon_normal)
            self.current_icon_state = 'normal'

        self.thread = QThread()
        self.worker = Worker(self.detection_model, self.detection_transform, self.device)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)
        self.worker.progress.connect(self.handle_progress)
        self.thread.start()
        self.toggle_action.setText("停止")
        self.tray_icon.setToolTip("疲労検出システム 監視中")

    def stop_monitoring(self):
        if self.worker:
            self.worker.is_running = False
        self.is_monitoring = False
        self.stopwatch_window.stop()
        self.toggl_manager.stop_session()
        self.toggle_action.setText("開始")
        self.tray_icon.setToolTip("疲労検出システム 停止中")

    def on_thread_finished(self):
        if self.thread:
            self.thread.deleteLater()
            self.thread = None
        self.worker = None

    def handle_progress(self, analysis_results):
        current_time = time.time()
        score = self.scorer.update_score(analysis_results)
        self.graph_window.update_graph(current_time, score)
        
        log_data = [
            datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S"),
            ','.join(analysis_results['detections']) if analysis_results['detections'] else 'none',
            score,
            analysis_results['perclos'],
            analysis_results['is_microsleep']
        ]
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_data)
        
        threshold = self.scorer.settings['fatigue_threshold']
        
        if score >= threshold and self.current_icon_state != 'fatigue':
            self.tray_icon.setIcon(self.icon_fatigue)
            self.current_icon_state = 'fatigue'
        elif score < threshold and self.current_icon_state != 'normal':
            self.tray_icon.setIcon(self.icon_normal)
            self.current_icon_state = 'normal'
        
        if score >= threshold and not self.warning_issued:
            self.notifier.issue_warning(score, reason="fatigue")
            self.warning_issued = True
            self.high_score_start_time = current_time
        elif score < threshold * 0.8:
            self.warning_issued = False
            self.high_score_start_time = None
            self.sustained_warning_issued = False

        if self.high_score_start_time and not self.sustained_warning_issued:
            if (current_time - self.high_score_start_time) > 300: # 5分
                self.notifier.issue_warning(score, reason="sustained_fatigue")
                self.sustained_warning_issued = True

        if not self.break_warning_issued:
            if self.stopwatch_window.elapsed_seconds > 5400: # 90分
                self.notifier.issue_warning(score, reason="duration")
                self.break_warning_issued = True

    def show_settings(self):
        if self.settings_window.exec():
            self.load_settings_and_init()

    def quit(self):
        if self.is_monitoring:
            self.stop_monitoring()
        super().quit()

# ================================================================
# アプリケーション実行
# ================================================================
if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Detections', 'FatigueScore', 'PERCLOS', 'IsMicrosleep'])

    app = FatigueMonitorApp(sys.argv)
    sys.exit(app.exec())