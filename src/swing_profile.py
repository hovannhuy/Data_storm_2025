import pandas as pd
from pathlib import Path

class SwingProfile:
    """
    Class để quản lý và cung cấp hồ sơ cú swing lý tưởng.
    """
    def __init__(self, feature_db_path):
        if not Path(feature_db_path).exists():
            raise FileNotFoundError(f"Feature database not found at: {feature_db_path}")
        
        df_features = pd.read_csv(feature_db_path)
        
        # Lọc ra các cú swing tốt và tính giá trị trung bình
        good_swings_df = df_features[df_features['quality'] == 'Good Swings']
        if good_swings_df.empty:
            raise ValueError("No 'Good Swings' found in the feature database to create a profile.")
            
        self.ideal_profile = good_swings_df.mean(numeric_only=True).to_dict()
        
        # Các đặc trưng sẽ được hiển thị trên video
        self.feature_names = [
            'left_arm_angle', 'right_arm_angle', 'left_knee_angle', 'right_knee_angle',
            'shoulders_inclination', 'hips_inclination', 'pelvis_angle'
        ]
        print("Ideal Swing Profile loaded successfully.")

    def get_ideal_profile(self):
        return self.ideal_profile

    def get_feature_names_for_display(self):
        return self.feature_names