# 標籤分類系統
在本次專案中，我們以研華科技提供的工業相機 **iCAM540** 為基礎，結合 **AWS** 雲端平台，開發出一套照片加工與資料自動化分類系統。透過影像強化與標籤分析技術，我們能有效提取產品資訊，進行分類與即期管理，提升企業對庫存及物流的掌握度，降低人力成本，並加快作業效率。
## 專案介紹
這個系統結合了 iCAM540 相機進行影像擷取，並透過物件偵測技術來進行自動拍照。在拍照後，系統會將影像自動上傳至 **AWS S3** 儲存空間。此專案旨在提高工業生產過程中的效率，並且能夠隨時掌握生產過程中的資訊，實現自動化數據管理。
## 主要功能
- 使用 iCAM540 工業相機進行物件偵測與拍照。
- 偵測到物體後，會自動進行照片拍攝。
- 照片在本地儲存後，會自動上傳至 AWS S3 儲存空間。
- 上傳過程中進行多次重試機制，以確保資料不丟失。
- 可設定物體穩定偵測時間與拍照間隔，確保影像品質。
## 安裝與使用
### 安裝依賴
請確保你的環境中已經安裝以下依賴：

1. **OpenCV**：用於影像處理與物體偵測。
2. **Boto3**：用於與 AWS S3 進行交互。

可以使用以下命令安裝這些依賴：
pip install opencv-python boto3

### src/camera.py
import cv2
import numpy as np
import time
from time import sleep
import boto3
from botocore.exceptions import NoCredentialsError
from datetime import datetime
import os

class ICam540Camera:
    def __init__(self):
        self.camera = None
        self.save_dir = "photos"
        self.min_detection_time = 2  # 偵測確認時間（秒）
        self.last_upload_time = 0
        self.upload_interval = 5  # 上傳間隔（秒）

    def initialize_camera(self):
        try:
            print("正在初始化 iCam540...")

            # 嘗試連接 iCam540
            self.camera = cv2.VideoCapture("/dev/video10")
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture("/dev/video1")

            if not self.camera.isOpened():
                print("無法連接 iCam540")
                return False

            # 設置攝像頭參數
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1408)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            print("iCam540 初始化成功")
            return True

        except Exception as e:
            print(f"相機初始化錯誤: {e}")
            return False

    def detect_objects(self, frame):
        try:
            # 轉換為灰度圖
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # 增加高斯模糊的核大小，減少噪點影響
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
            # 調整Canny邊緣檢測的閾值
            edges = cv2.Canny(blurred, 20, 100)  # 降低閾值使其更容易檢測邊緣
        
            # 進行形態學操作以連接斷開的邊緣
            kernel = np.ones((5,5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel, iterations=1)
        
            # 尋找輪廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            detected_objects = []
            max_area = 0
            min_area = 500  # 降低最小面積閾值
        
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:  # 降低面積閾值
                    x, y, w, h = cv2.boundingRect(contour)
                    # 過濾太細長的區域
                    aspect_ratio = float(w)/h
                    if 0.2 < aspect_ratio < 5:  # 調整長寬比例的限制
                        detected_objects.append({
                            'box': (x, y, w, h),
                            'area': area
                        })
                        max_area = max(max_area, area)
        
            if detected_objects:
                print(f"偵測到 {len(detected_objects)} 個物體, 最大面積 {max_area:.0f}")
                sleep(1)
            else:
                print("未偵測到物體")
                # 輸出當前最大輪廓的面積，幫助調試
                if contours:
                    max_contour_area = max(cv2.contourArea(c) for c in contours)
                    print(f"最大輪廓面積: {max_contour_area:.0f}")
                sleep(1)
            return detected_objects

        except Exception as e:
            print(f"物體偵測錯誤: {e}")
            return []

    def capture_and_upload(self):
        if not self.initialize_camera():
            return

        print("開始自動偵測...")
        print("按 Ctrl+C 退出")

        last_detect_count = 0
        stable_start_time = None

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("無法獲取影像")
                    break

                detected_objects = self.detect_objects(frame)
                current_detect_count = len(detected_objects)
                current_time = time.time()

                if current_detect_count > 0:
                    if current_detect_count == last_detect_count:
                        if stable_start_time is None:
                            stable_start_time = current_time
                            print("開始計時！")
                        elif (current_time - stable_start_time) >= self.min_detection_time:
                            if (current_time - self.last_upload_time) >= self.upload_interval:
                                print("偵測穩定超過2秒，可以拍照了！")

                                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                                local_filename = f"icam540_{timestamp}.jpg"
                                s3_key = f"parts_detection/{local_filename}"

                                saved = cv2.imwrite(os.path.join(self.save_dir, local_filename), frame)
                                if saved:
                                    print(f"照片儲存成功: {local_filename}")
                                else:
                                    print(f"照片儲存失敗: {local_filename}")

                                if upload_to_s3(os.path.join(self.save_dir, local_filename), s3_key):
                                    print(f"圖片已上傳至 S3")
                                    self.last_upload_time = current_time
                                else:
                                    # 如果上傳失敗，保存到備份目錄
                                    backup_dir = "failed_uploads"
                                    if not os.path.exists(backup_dir):
                                        os.makedirs(backup_dir)
                                    import shutil
                                    backup_path = os.path.join(backup_dir, local_filename)
                                    shutil.move(os.path.join(self.save_dir, local_filename), backup_path)
                                    print(f"上傳失敗，文件已移至: {backup_path}")
                                    continue

                                os.remove(os.path.join(self.save_dir, local_filename))
                                print(f"本地檔案已刪除")

                                stable_start_time = None  # 重置
                    else:
                        stable_start_time = current_time
                else:
                    stable_start_time = None

                last_detect_count = current_detect_count
                time.sleep(0.1)  # 降低 CPU 使用率

        except KeyboardInterrupt:
            print("\n程式停止運行")
        except Exception as e:
            print(f"錯誤: {str(e)}")
        finally:
            self.camera.release()

def upload_to_s3(local_file, s3_key, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"嘗試上傳第 {attempt + 1} 次...")
            
            # 修正 boto3 配置
            s3_client = boto3.client(
                's3',
                region_name='us-west-2',
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
            )


            bucket_name = 'awsntu'
            
            # 檢查文件是否存在
            if not os.path.exists(local_file):
                print(f"錯誤：找不到檔案 {local_file}")
                return False

            # 上傳文件
            with open(local_file, 'rb') as file_data:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=file_data
                )
            
            print(f"上傳成功: s3://{bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            print(f"上傳嘗試 {attempt + 1} 失敗: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
            else:
                print("已達到最大重試次數")
                return False
    
    return False


if __name__ == "__main__":
    # 設置環境變量
    os.environ['AWS_ACCESS_KEY_ID'] = 'ASIA2WCWFYZUXIP5XISV'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'BvJUZqDANTSusLaSYiHAwgjt3j7lr/2bPehV52kV'
    os.environ['AWS_SESSION_TOKEN'] = 'IQoJb3JpZ2luX2VjELv//////////wEaCXVzLWVhc3QtMSJIMEYCIQDLGZ1PLuC1ooms60DwoL7dcyqSM7itUDolL+udqVvPLAIhAO5NocT+Ldcd7FtbbOGEljub/D/aOtAIgzerticBXqYDKpkCCFQQARoMNzM0NjIwMTQxMTYxIgynZPA7OJLwejwzRlcq9gHCh2LKUBhW9uLSWSclmJf9eYiu4OG+YH/otx7I6bV1fXJmi2UeSgKcbZByDCCFS9A9jyrBICJJFXQWi05ZDP4yk9X7HLs25nEt6XciTf3JGkExAYU3/SLkckmCLXwOGOTVT/fCsHaVs7vjYU9t0103cuEwlspYJ3R8rAN5BX6EjazuT1c8psInMC5AySab8GEhE9iHRHDg70mYPUkxkXGO9i+gQ83JPPeTz1vRc3S5Pl1mQ5h1E1VmLeLTGE0vulaCy+uZlsTt6KbinRvZSYk77nGo9N9ExKRVA+HjpT8cx9APWzWUrPrTv5HFMWxXBAd0TW/5bKkw0Ly2wAY6nAFahREHrDXYpdSuux/P1PrReKL7Jb6L7jmabH1gOD+1RR3Qhp6tDLxTYSpSGw3Db23HxmQjtHnx/+IhsWoqnXvwV/MhrJhaTTYcJpu5xyBrIeN/1pq+/5xKl8ujHMqksMswd9izRv9vvb9SSzgF5LtvjxS+4IEtf15oUdQmq5eu9FKbp+UR6sxSgeOc6mjNSlQg6V2vnyWXvoCD8Z4='
    os.environ['S3_BUCKET'] = 'awsntu'

    # 創建必要的目錄
    for directory in ['photos', 'failed_uploads']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"創建目錄: {directory}")

    camera = ICam540Camera()
    camera.min_detection_time = 2   # 穩定2秒才拍照
    camera.upload_interval = 5      # 每次拍照間隔至少5秒
    camera.capture_and_upload()

### src/detect_text.py
!pip install opencv-python-headless
!pip install Pillow
!pip install openpyxl

class LabelConfig:
    LABEL_FIELDS = {
        'product_name': {
            'keywords': ['品名', '產品名稱', '商品名',"LABEL"],
            'required': True,
            'description': '產品名稱'
        },
        'production_date': {
            'keywords': ['製造日期', '製造日', '生產日期','production_date'],
            'required': True,
            'description': '製造日期'
        },
        'expiry_date': {
            'keywords': ['有效期限', '保存期限', '效期',"expiry_date"],
            'required': True,
            'description': '有效期限'
        },
        'batch_number': {
            'keywords': ['批號', '批次', 'lot no', 'batch no'],
            'required': True,
            'description': '批號'
        },
        'manufacturer': {
            'keywords': ['製造商', '製造廠商', '製造工廠',"manufacturer"],
            'required': True,
            'description': '製造商'
        },
        'storage_condition': {
            'keywords': ['儲存條件', '保存條件', '儲存方式',"storage_condition"],
            'required': True,
            'description': '儲存條件'
        },
        'weight': {
            'keywords': ['重量', '淨重', '容量', '內容量',"weight"],
            'required': True,
            'description': '重量/容量'
        },
        'ingredients': {
            'keywords': ['成分', '內容物', '配料',"ingredients"],
            'required': True,
            'description': '成分'
        },
        'origin': {
            'keywords': ['產地', '原產地', '生產地','origin'],
            'required': True,
            'description': '產地'
        },
        'package_type': {
            'keywords': ['包裝', '包裝方式', '包裝類型','package_type'],
            'required': True,
            'description': '包裝類型'
        }
    }

    @classmethod
    def get_field_names(cls):
        return list(cls.LABEL_FIELDS.keys())

    @classmethod
    def get_required_fields(cls):
        return [field for field, config in cls.LABEL_FIELDS.items() 
                if config['required']]

    @classmethod
    def get_field_description(cls, field_name):
        return cls.LABEL_FIELDS.get(field_name, {}).get('description', '')

    @classmethod
    def get_field_keywords(cls, field_name):
        return cls.LABEL_FIELDS.get(field_name, {}).get('keywords', [])
import pandas as pd
pd.set_option('display.max_rows', None)  # 顯示所有行
pd.set_option('display.max_columns', None)  # 顯示所有列
pd.set_option('display.width', None)  # 自動調整顯示寬度
pd.set_option('display.max_colwidth', None)  # 顯示完整的列內容
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

class ResultManager:
    def __init__(self, s3_client):
        self.s3_client = s3_client
        self.master_file_name = 'processing_history.xlsx'
        self.local_path = "/home/sagemaker-user/output"
        self.master_file_path = os.path.join(self.local_path, self.master_file_name)

    def _create_new_master_file(self, df, sheet_name):
        """建立新的主檔案，加入錯誤處理"""
        try:
            # 創建一個基本的 Excel 檔案
            with pd.ExcelWriter(self.master_file_path, engine='openpyxl') as writer:
                # 寫入資料
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                # 創建摘要表
                summary_df = pd.DataFrame({
                    'Batch': [sheet_name],
                    'Total Records': [len(df)],
                    'Success Records': [df.notna().sum().mean()],
                    'Success Rate': [(df.notna().sum() / len(df)).mean() * 100],
                    'Processing Date': [sheet_name.split('_')[1]]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"成功創建新的主檔案: {self.master_file_path}")
            return True
        except Exception as e:
            print(f"創建主檔案時發生錯誤: {str(e)}")
            return False

    def save_and_update_results(self, df, bucket_name, source_prefix):
        """儲存新結果並更新主檔案"""
        if df is None or df.empty:
            print("沒有資料需要儲存")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sheet_name = f'Batch_{timestamp}'

        try:
            # 先將結果儲存為 CSV 作為備份
            csv_backup_path = os.path.join(self.local_path, f'backup_{timestamp}.csv')
            df.to_csv(csv_backup_path, index=False)
            print(f"已建立備份檔案: {csv_backup_path}")

            # 嘗試建立或更新 Excel 檔案
            if not os.path.exists(self.master_file_path):
                success = self._create_new_master_file(df, sheet_name)
                if not success:
                    raise Exception("無法創建主檔案")
            else:
                try:
                    # 讀取現有檔案
                    with pd.ExcelWriter(self.master_file_path, engine='openpyxl', mode='a') as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"已更新主檔案，新增工作表: {sheet_name}")
                except Exception as e:
                    print(f"更新現有檔案失敗，嘗試重新創建: {str(e)}")
                    success = self._create_new_master_file(df, sheet_name)
                    if not success:
                        raise Exception("無法重新創建主檔案")

            # 上傳到 S3
            try:
                # 上傳 Excel 檔案
                self.s3_client.upload_file(
                    self.master_file_path,
                    bucket_name,
                    f'master/{self.master_file_name}'
                )
                print(f"已上傳主檔案至 S3: s3://awsntu/master/{self.master_file_name}")

                # 上傳備份 CSV
                self.s3_client.upload_file(
                    csv_backup_path,
                    bucket_name,
                    f'master/backups/backup_{timestamp}.csv'
                )
                print(f"已上傳備份檔案至 S3: s3://awsntu/master/backups/backup_{timestamp}.csv")

            except Exception as e:
                print(f"上傳到 S3 時發生錯誤: {str(e)}")

        except Exception as e:
            print(f"處理結果時發生錯誤: {str(e)}")
            print("資料已保存在備份 CSV 檔案中")

    def _update_summary_sheet(self):
        """更新摘要表"""
        try:
            # 讀取所有工作表
            excel_file = pd.ExcelFile(self.master_file_path)
            sheet_names = [name for name in excel_file.sheet_names if name != 'Summary']
            
            summary_data = []
            for sheet in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet)
                summary = {
                    'Batch': sheet,
                    'Total Records': len(df),
                    'Success Records': df.notna().sum().mean(),
                    'Success Rate': (df.notna().sum() / len(df)).mean() * 100,
                    'Processing Date': sheet.split('_')[1]
                }
                summary_data.append(summary)
            
            summary_df = pd.DataFrame(summary_data)
            
            # 使用新的 ExcelWriter 來更新摘要
            with pd.ExcelWriter(self.master_file_path, engine='openpyxl') as writer:
                for sheet in sheet_names:
                    # 複製原有的資料
                    temp_df = pd.read_excel(excel_file, sheet_name=sheet)
                    temp_df.to_excel(writer, sheet_name=sheet, index=False)
                # 寫入更新的摘要
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
        except Exception as e:
            print(f"更新摘要表時發生錯誤: {str(e)}")

class ImagePreprocessor:
    @staticmethod
    def enhance_image(image_bytes):
        """綜合圖片增強處理"""
        # 將 bytes 轉換為 numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. 調整大小（如果圖片太小）
        min_height = 1000
        height, width = image.shape[:2]
        if height < min_height:
            scale = min_height / height
            image = cv2.resize(image, None, fx=scale, fy=scale, 
                             interpolation=cv2.INTER_CUBIC)

        # 2. 灰度轉換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 3. 降噪
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 4. 自適應二值化
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. 邊緣增強
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(binary, -1, kernel)
        
        # 將處理後的圖片轉回 bytes
        success, encoded_image = cv2.imencode('.png', sharpened)
        return encoded_image.tobytes()

    @staticmethod
    def enhance_image_pil(image_bytes):
        """使用 PIL 進行圖片增強"""
        # 將 bytes 轉換為 PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # 1. 調整對比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # 增加對比度
        
        # 2. 調整銳利度
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # 增加銳利度
        
        # 3. 調整亮度
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)  # 稍微增加亮度
        
        # 將處理後的圖片轉回 bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    @staticmethod
    def deskew_image(image_bytes):
        """校正圖片傾斜"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 檢測邊緣
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫變換檢測直線
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            # 計算平均角度
            angles = []
            for rho, theta in lines[0]:
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                median_angle = np.median(angles)
                
                # 旋轉圖片
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
                
                # 將處理後的圖片轉回 bytes
                success, encoded_image = cv2.imencode('.png', rotated)
                return encoded_image.tobytes()
        
        return image_bytes
def setup_sagemaker():
    try:
        role = sagemaker.get_execution_role()
        session = sagemaker.Session()
        region = session.boto_region_name
        return role, session, region
    except Exception as e:
        print(f"設定 SageMaker 時發生錯誤: {str(e)}")
        return None, None, None

def init_aws_clients():
    try:
        s3_client = boto3.client('s3')
        textract_client = boto3.client('textract')
        return s3_client, textract_client
    except Exception as e:
        print(f"初始化 AWS 客戶端時發生錯誤: {str(e)}")
        return None, None
class ImageProcessor:
    def __init__(self, s3_client, textract_client):
        self.s3_client = s3_client
        self.textract_client = textract_client
        self.output_path = "/home/sagemaker-user/output"
        self.result_manager = ResultManager(s3_client)
        os.makedirs(self.output_path, exist_ok=True)

    def parse_label_text(self, blocks):
        #print("==== OCR 偵測到的所有行 ====")
        #for idx, line in enumerate(text_lines):
            #print(f"{idx}: {line}")
        #print("============================")

        product_info = {
            'product_name': None,
            'production_date': None,
            'expiry_date': None
        }

        if not blocks:
            return product_info
    
        # 先把所有 LINE 的文字抓出來
        lines = [block['Text'] for block in blocks if block['BlockType'] == 'LINE']
    
        # 建立一個完整的大字串，方便搜尋
        full_text = '\n'.join(lines).lower()
    
        # 然後用模糊規則搜尋
        for idx, line in enumerate(lines):
            line_lower = line.lower()
    
            if any(keyword in line_lower for keyword in ['品名', '產品名稱', '商品名',"LABEL"]):
                product_info['product_name','label'] = self.extract_value(line, lines, idx)
    
            elif any(keyword in line_lower for keyword in ['製造日期', '製造日', '生產日期','production_date']):
                product_info['production_date'] = self.extract_value(line, lines, idx)

            elif any(keyword in line_lower for keyword in ['有效期限', '保存期限', '效期',"expiry_date"]):
                product_info['expiry_date'] = self.extract_value(line, lines, idx)

            elif any(keyword in line_lower for keyword in ['批號', '批次', 'lot no', 'batch no']):
                product_info['batch_number'] = self.extract_value(line, lines, idx)
                
            elif any(keyword in line_lower for keyword in ['製造商', '製造廠商', '製造工廠',"manufacturer"]):
                product_info['manufacturer'] = self.extract_value(line, lines, idx)
                
            elif any(keyword in line_lower for keyword in ['儲存條件', '保存條件', '儲存方式',"storage_condition"]):
                product_info['storage_condition'] = self.extract_value(line, lines, idx)
                
            elif any(keyword in line_lower for keyword in ['重量', '淨重', '容量', '內容量',"weight"]):
                product_info['weight'] = self.extract_value(line, lines, idx)
                
            elif any(keyword in line_lower for keyword in ['成分', '內容物', '配料',"ingredients"]):
                product_info['ingredients'] = self.extract_value(line, lines, idx)
                
            elif any(keyword in line_lower for keyword in ['產地', '原產地', '生產地','origin']):
                product_info['origin'] = self.extract_value(line, lines, idx)
                
            elif any(keyword in line_lower for keyword in ['包裝', '包裝方式', '包裝類型','package_type']):
                product_info['package_type'] = self.extract_value(line, lines, idx)

        return product_info

    def extract_value(self, line, lines, idx):
        # 嘗試從這一行抓值
        if '：' in line:
            return line.split('：')[-1].strip()
        elif ':' in line:
            return line.split(':')[-1].strip()
        else:
            # 如果這一行沒有，可能值在下一行
            if idx + 1 < len(lines):
                return lines[idx + 1].strip()
        return None
    
    def get_text_from_image(self, bucket, key):
        """從圖片中提取文字"""
        try:
            response = self.textract_client.detect_document_text(
                Document={
                    'S3Object': {
                        'Bucket': bucket,
                        'Name': key
                    }
                }
            )
            return response.get('Blocks', [])
        except Exception as e:
            print(f"處理圖片時發生錯誤 ({key}): {str(e)}")
            return None

    def process_images(self, bucket_name, prefix=''):
        """處理多張圖片"""
        try:
            kwargs = {'Bucket': bucket_name}
            if prefix:
                kwargs['Prefix'] = prefix
            
            response = self.s3_client.list_objects_v2(**kwargs)
            
            if 'Contents' not in response:
                print(f"在 bucket awsntu 中未找到圖片")
                return None

            results = []
            image_files = [obj for obj in response['Contents'] 
                         if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            total_images = len(image_files)
            print(f"找到 {total_images} 張圖片需要處理")

            for idx, obj in enumerate(image_files, 1):
                print(f"處理圖片 {idx}/{total_images}: {obj['Key']}")
                
                blocks = self.get_text_from_image(bucket_name, obj['Key'])
                if blocks:
                    product_info = self.parse_label_text(blocks)
                    product_info['image_name'] = obj['Key']
                    results.append(product_info)
                
                time.sleep(0.5)

            return pd.DataFrame(results) if results else None

        except Exception as e:
            print(f"處理圖片時發生錯誤: {str(e)}")
            return None

    def save_results(self, df, bucket_name, source_prefix):
        """儲存處理結果"""
        if df is None or df.empty:
            print("沒有資料需要儲存")
            return

        # 使用 ResultManager 儲存結果
        self.result_manager.save_and_update_results(df, bucket_name, source_prefix)


def main():
    role, session, region = setup_sagemaker()
    s3_client, textract_client = init_aws_clients()
    
    if None in (s3_client, textract_client):
        print("初始化失敗，程式終止")
        return

    processor = ImageProcessor(s3_client, textract_client)
    
    BUCKET_NAME = 'awsntu'  # 改為你的 bucket 名稱
    PREFIX = 'parts_detection/'  # 改為你的圖片所在資料夾路徑
    
    print("開始處理圖片...")
    df = processor.process_images(BUCKET_NAME, PREFIX)
    
    if df is not None:
        processor.save_results(df, BUCKET_NAME, PREFIX)  
        
        print("\n處理結果摘要:")
        print(f"總處理圖片數: {len(df)}")
        print("\n範例資料:")
        print(df.head())
    else:
        print("處理過程中沒有產生有效的資料")


if __name__ == "__main__":
    main()

### 圖片
