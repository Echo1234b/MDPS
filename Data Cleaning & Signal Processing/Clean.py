# Clean.py
import pandas as pd
import numpy as np
from data_processing import DataCleaningPipeline

# --- إعدادات التجربة ---
# في مشروع حقيقي، سيتم استيراد هذا من ملف config.py
class MockConfig:
    class QA: missing_handler = {'strategy': 'ffill'}; outlier_detector = {}
    class TSA: timestamp_normalizer = {}; freq_converter = {'freq': '1H'}
    class NST: data_smoother = {'ema_span': 14}; volume_normalizer = {'method': 'robust'}
    qa = QA(); tsa = TSA(); nst = NST()
config = MockConfig()
# --- نهاية الإعدادات ---

# إنشاء بيانات تجريبية
data = {
    'time': np.arange(1609459200, 1609459200 + 100 * 3600, 3600), # 100 ساعة
    'open': np.random.rand(100) * 10 + 100,
    'close': np.random.rand(100) * 10 + 100,
    'high': np.random.rand(100) * 5 + 110,
    'low': np.random.rand(100) * 5 + 95,
    'volume': np.random.randint(1000, 5000, 100)
}
your_dataframe = pd.DataFrame(data)

print("--- Starting a dynamic data processing pipeline ---")

# إنشاء نسخة من خط الأنابيب مع الإعدادات
pipeline = DataCleaningPipeline(config=config)

# بناء وتشغيل خط الأنابيب بسلاسة باستخدام الواجهة المرنة
final_data = (pipeline
              .start_with(your_dataframe)
              .sanitize_data()
              .normalize_timestamps()
              .smooth_data(columns=['close', 'open'])
              .normalize_volume(columns=['volume'])
              .get_dataframe()
             )

print("\nProcessing Finished!")
print("\nFinal Data Head:")
print(final_data.head())
print(f"\nFinal Data Shape: {final_data.shape}")
print("\nFinal Columns:", final_data.columns.tolist())