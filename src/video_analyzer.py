import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import mediapipe as mp
from pathlib import Path

from .pose_extractor import PoseFeatureExtractor # Import tương đối

class VideoAnalyzer:
    """
    Class chính để tạo video phân tích.
    """
    def __init__(self, swing_profile):
        self.extractor = PoseFeatureExtractor()
        self.profile = swing_profile
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.feature_names = self.profile.get_feature_names_for_display()

    def generate_analysis_video(self, video_path, output_path, slowdown_factor=1.4):
        ideal_profile = self.profile.get_ideal_profile()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        output_fps = fps / slowdown_factor
        
        out_width, out_height = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (out_width, out_height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(total_frames), desc=f"Analyzing {Path(video_path).name}"):
            ret, frame = cap.read()
            if not ret: break

            composite_frame = self._create_composite_frame(frame, ideal_profile)
            out.write(composite_frame)

        cap.release(); out.release()
        print(f"\nAnalysis video saved to: {output_path}")

    def _create_composite_frame(self, frame, ideal_profile):
        panel_width, panel_height = 640, 360
        
        original_panel = cv2.resize(frame, (panel_width, panel_height))
        pose_panel = original_panel.copy()
        
        results = self.extractor.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        features = {}
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                pose_panel, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            features = self.extractor._calculate_geometric_features(results.pose_landmarks, frame.shape)

        table_panel = self._create_feature_table_panel(features, ideal_profile)
        radar_panel = self._create_radar_chart_panel(features, ideal_profile)
        
        top_row = np.concatenate((original_panel, pose_panel), axis=1)
        bottom_row = np.concatenate((table_panel, radar_panel), axis=1)
        return np.concatenate((top_row, bottom_row), axis=0)

    def _create_feature_table_panel(self, features, ideal_profile):
        panel = np.ones((360, 640, 3), dtype=np.uint8) * 20
        y_pos = 40
        for name in self.feature_names:
            text = f"{name.replace('_', ' ').title()}:"
            cv2.putText(panel, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # --- CẬP NHẬT: Chỉ hiển thị nếu feature được tính toán ---
            if name in features:
                val_text = f"{features[name]:.1f}"
                cv2.putText(panel, val_text, (400, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_pos += 40
        return panel
    
    def _create_radar_chart_panel(self, features, ideal_profile):
        labels = [name.replace('_', '\n').title() for name in self.feature_names]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
        
        fig, ax = plt.subplots(figsize=(6.4, 3.6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#141414'); ax.set_facecolor('#141414')

        ideal_values = [100] * num_vars + [100]
        # --- CẬP NHẬT: Gán giá trị 0 cho các feature bị thiếu ---
        current_values = [np.clip((features.get(f, 0) / (ideal_profile.get(f, 1) or 1)) * 100, 0, 150) for f in self.feature_names] + [np.clip((features.get(self.feature_names[0], 0) / (ideal_profile.get(self.feature_names[0], 1) or 1)) * 100, 0, 150)]
        
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, color='white', fontsize=8)
        ax.set_rlabel_position(0); ax.tick_params(axis='y', colors='gray')
        ax.plot(angles, ideal_values, color='cyan', linewidth=1, linestyle='dashed')
        ax.fill(angles, ideal_values, 'cyan', alpha=0.1)
        ax.plot(angles, current_values, color='orange', linewidth=2)
        ax.fill(angles, current_values, 'orange', alpha=0.25)
        plt.yticks([50, 100, 150], ["50%", "100%", "150%"], color="gray", size=7)
        plt.ylim(0, 160); ax.grid(color='gray', linestyle='--', linewidth=0.5)
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)