import logging
import uuid
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không tương tác để tránh lỗi tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến toàn cục để lưu model và các thành phần
model = None
X_test = None
y_test = None
df = None
scaler = None
main_door_encoder = None
balcony_encoder = None
legal_doc_encoder = None

# Hàm phân tích giá
def parse_price(price):
    if pd.isna(price):
        return np.nan
    price = str(price).lower().replace(',', '.').replace(' ', '')
    try:
        if 'tỷ' in price:
            price = price.replace('tỷ', '')
            return float(price) * 1e9
        elif 'triệu' in price:
            price = price.replace('triệu', '')
            return float(price) * 1e6
        else:
            return float(price)
    except (ValueError, TypeError):
        return np.nan

# Hàm trích xuất diện tích
def extract_area(area):
    if pd.isna(area):
        return np.nan
    area = str(area).lower().replace(',', '.')
    try:
        match = re.search(r'(\d+\.?\d*)', area)
        if match:
            return float(match.group(1))
        return np.nan
    except (ValueError, TypeError):
        return np.nan

# Hàm trích xuất số phòng
def extract_room(room):
    if pd.isna(room) or str(room).strip() == '':
        return np.nan
    room = str(room).lower()
    try:
        match = re.search(r'(\d+)', room)
        if match:
            return float(match.group(1))
        return np.nan
    except (ValueError, TypeError):
        return np.nan

# Hàm trích xuất số tầng
def extract_floors(floors):
    if pd.isna(floors):
        return np.nan
    floors = str(floors).lower()
    try:
        match = re.search(r'(\d+)', floors)
        if match:
            return float(match.group(1))
        return np.nan
    except (ValueError, TypeError):
        return np.nan

# Hàm mã hóa loại nhà
def encode_category(category):
    if pd.isna(category):
        category = 'Unknown'
    category = str(category).lower().strip()
    categories = {
        'nhà hẻm, ngõ': [1, 0, 0, 0, 0, 0, 0, 0],
        'nhà mặt tiền': [0, 1, 0, 0, 0, 0, 0, 0],
        'biệt thự, villa': [0, 0, 1, 0, 0, 0, 0, 0],
        'đất thổ cư': [0, 0, 0, 1, 0, 0, 0, 0],
        'nhà phố liền kề': [0, 0, 0, 0, 1, 0, 0, 0],
        'căn hộ chung cư': [0, 0, 0, 0, 0, 1, 0, 0],
        'khách sạn, nhà hàng': [0, 0, 0, 0, 0, 0, 1, 0],
        'unknown': [0, 0, 0, 0, 0, 0, 0, 1]
    }
    return categories.get(category, [0, 0, 0, 0, 0, 0, 0, 1])

# Hàm mã hóa vị trí (quận)
def encode_location(loc):
    if not loc or pd.isna(loc) or loc.lower().strip() == 'unknown':
        logger.debug(f"Vị trí không hợp lệ hoặc không xác định: {loc}, sử dụng district_code=0")
        return 0
    loc = loc.lower().strip()
    districts = {
        "bình thạnh": 1, "gò vấp": 2, "tân bình": 3, "quận 1": 4, "quận 3": 5, "quận 10": 6,
        "thủ đức": 7, "phú nhuận": 8, "quận 7": 9, "bình tân": 10, "nhà bè": 11, "củ chi": 12,
        "hóc môn": 13, "quận 12": 14, "quận 8": 15, "quận 5": 16, "quận 6": 17, "quận 9": 18,
        "quận 2": 19, "quận 4": 20, "bình chánh": 21, "tp. thủ đức - quận thủ đức": 22,
        "tp. thủ đức - quận 9": 23, "tp. thủ đức - quận 2": 24, "cần giờ": 25, "quận 11": 26,
        "tân phú": 27
    }
    district_code = 0
    for name, code in districts.items():
        if name in loc:
            district_code = code
            break
    if district_code == 0:
        logger.warning(f"Không tìm thấy quận trong vị trí: {loc}, sử dụng district_code=0")
    return district_code

# Hàm mã hóa hướng
def encode_direction(direction, encoder):
    if not direction or pd.isna(direction) or str(direction).lower().strip() == '':
        direction = 'unknown'
    direction = str(direction).lower().strip()
    direction_array = np.array([[direction]])
    encoded = encoder.transform(direction_array)[0]
    return encoded.tolist()

# Hàm mã hóa giấy tờ pháp lý
def encode_legal_documents(doc, encoder):
    if not doc or pd.isna(doc) or str(doc).lower().strip() == '':
        doc = 'unknown'
    doc = str(doc).lower().strip()
    doc_array = np.array([[doc]])
    encoded = encoder.transform(doc_array)[0]
    return encoded.tolist()

# Hàm khởi tạo dữ liệu và mô hình
def initialize_data_and_model(file_path="real_estate_listings.csv"):
    global model, X_test, y_test, df, scaler, main_door_encoder, balcony_encoder, legal_doc_encoder

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset được tải thành công: {file_path}, {len(df)} mẫu")
    except FileNotFoundError:
        logger.error(f"File {file_path} không tồn tại")
        raise FileNotFoundError(f"File {file_path} không tồn tại")

    # Thêm ID duy nhất
    df['ID'] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Làm sạch ban đầu
    df = df.dropna(subset=['Price', 'Land Area'])
    logger.info(f"Số mẫu sau khi xóa NaN ở Price và Land Area: {len(df)}")

    # Phân tích đặc trưng
    df['Price'] = df['Price'].apply(parse_price)
    df['Land Area'] = df['Land Area'].apply(extract_area)

    # Loại bỏ giá trị không hợp lệ
    df = df.dropna(subset=['Price', 'Land Area'])
    logger.info(f"Số mẫu sau khi xóa Price và Land Area không hợp lệ: {len(df)}")

    # Phân tích Bedrooms và Toilets
    df['Bedrooms'] = df.apply(
        lambda x: 0 if 'Đất' in str(x['Type of House']) else extract_room(x['Bedrooms']), axis=1
    )
    df['Toilets'] = df.apply(
        lambda x: 0 if 'Đất' in str(x['Type of House']) else extract_room(x['Toilets']), axis=1
    )

    # Điền giá trị thiếu
    df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median(skipna=True))
    df['Toilets'] = df['Toilets'].fillna(df['Toilets'].median(skipna=True))
    df['Total Floors'] = df['Total Floors'].apply(extract_floors).fillna(1)
    df['Type of House'] = df['Type of House'].fillna('Unknown')
    df['Location'] = df['Location'].fillna('Unknown')
    df['Main Door Direction'] = df['Main Door Direction'].fillna('unknown')
    df['Balcony Direction'] = df['Balcony Direction'].fillna('unknown')
    df['Legal Documents'] = df['Legal Documents'].fillna('unknown')

    # Thêm giá trên mỗi m²
    df['Price_per_m2'] = df['Price'] / df['Land Area']

    # Loại bỏ trùng lặp
    duplicates = df[df.duplicated(subset=['Price', 'Land Area', 'Location', 'Type of House', 'Bedrooms'])]
    logger.info(f"Số mẫu trùng lặp: {len(duplicates)}")
    duplicates.to_csv("duplicates.csv", index=False)
    logger.info("Đã lưu các mẫu trùng lặp vào duplicates.csv")
    df = df.drop_duplicates(subset=['Price', 'Land Area', 'Location', 'Type of House', 'Bedrooms'])
    logger.info(f"Số mẫu sau khi xóa trùng lặp: {len(df)}")

    # Loại bỏ ngoại lệ
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Price'] >= Q1 - 6 * IQR) & (df['Price'] <= Q3 + 6 * IQR)]
    df = df[(df['Price'] > 1e9) & (df['Land Area'] > 0)]
    logger.info(f"Số mẫu sau khi loại bỏ ngoại lệ: {len(df)}")

    # Biến đổi log cho Price
    df['Price'] = np.log1p(df['Price'])

    # Lưu dữ liệu đã xử lý
    df.to_csv("processed_real_estate_listings.csv", index=False)

    # Tạo trực quan hóa
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Price'].dropna(), bins=40)
    plt.title("Phân bố Giá Nhà (Logarit)")
    plt.xlabel("Log(Giá) (VND)")
    plt.ylabel("Số lượng")
    plt.savefig("price_distribution_log.png")
    plt.close()
    logger.info("Đã tạo price_distribution_log.png")

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['Land Area'].dropna())
    plt.title("Boxplot Diện tích đất")
    plt.savefig("land_area_boxplot.png")
    plt.close()
    logger.info("Đã tạo land_area_boxplot.png")

    plt.figure(figsize=(8, 4))
    df['Type of House'].value_counts().plot(kind='bar')
    plt.title("Số lượng theo loại nhà")
    plt.ylabel("Số căn")
    plt.savefig("house_type_bar.png")
    plt.close()
    logger.info("Đã tạo house_type_bar.png")

    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=df['Land Area'], y=df['Price'])
    plt.title("Diện tích đất vs. Log(Giá)")
    plt.savefig("land_area_vs_price_log.png")
    plt.close()
    logger.info("Đã tạo land_area_vs_price_log.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(df['Price_per_m2'].dropna(), bins=40)
    plt.title("Phân bố Giá trên mỗi m²")
    plt.xlabel("Giá trên mỗi m² (VND)")
    plt.ylabel("Số lượng")
    plt.savefig("price_per_m2_distribution.png")
    plt.close()
    logger.info("Đã tạo price_per_m2_distribution.png")

    # Khởi tạo các bộ mã hóa
    scaler = StandardScaler()
    main_door_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[[
        'bắc', 'nam', 'tây', 'tây bắc', 'tây nam', 'đông', 'đông bắc', 'đông nam', 'unknown'
    ]])
    balcony_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[[
        'bắc', 'nam', 'tây', 'tây bắc', 'tây nam', 'đông', 'đông bắc', 'đông nam', 'unknown'
    ]])
    legal_doc_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[[
        'giấy tờ hợp lệ', 'giấy tờ khác', 'hợp đồng mua bán', 'sổ hồng', 'sổ đỏ', 'đang chờ sổ', 'unknown'
    ]])

    # Chuẩn bị dữ liệu huấn luyện
    X_train, y_train = [], []
    expected_feature_count = 5 + 8 + 1 + 9 + 9 + 7  # 5 số + 8 loại nhà + 1 quận + 9 hướng cửa + 9 hướng ban công + 7 giấy tờ
    logger.info(f"Số đặc trưng dự kiến: {expected_feature_count}")

    # Fit các bộ mã hóa
    main_door_encoder.fit(df['Main Door Direction'].str.lower().fillna('unknown').values.reshape(-1, 1))
    balcony_encoder.fit(df['Balcony Direction'].str.lower().fillna('unknown').values.reshape(-1, 1))
    legal_doc_encoder.fit(df['Legal Documents'].str.lower().fillna('unknown').values.reshape(-1, 1))

    for _, row in df.iterrows():
        area = row['Land Area']
        bedrooms = row['Bedrooms']
        toilets = row['Toilets']
        floors = row['Total Floors']
        price_per_m2 = row['Price_per_m2']
        price = row['Price']
        category_vec = encode_category(row['Type of House'])
        district_code = encode_location(row['Location'])
        main_door_vec = encode_direction(row['Main Door Direction'], main_door_encoder)
        balcony_vec = encode_direction(row['Balcony Direction'], balcony_encoder)
        legal_doc_vec = encode_legal_documents(row['Legal Documents'], legal_doc_encoder)

        features = (
            [area, bedrooms, toilets, floors, price_per_m2] +
            category_vec +
            [district_code] +
            main_door_vec +
            balcony_vec +
            legal_doc_vec
        )
        if len(features) != expected_feature_count:
            logger.error(f"Chiều dài đặc trưng không đồng nhất: {len(features)} (dự kiến {expected_feature_count}) for location: {row['Location']}")
            continue
        X_train.append(features)
        y_train.append(price)

    if not X_train:
        raise ValueError("Không có mẫu huấn luyện hợp lệ sau khi tạo đặc trưng")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    logger.info(f"Dataset huấn luyện đã tạo với {len(X_train)} mẫu, {X_train.shape[1]} đặc trưng")

    # Chuẩn hóa đặc trưng số
    scaler.fit(X_train[:, :5])
    X_train[:, :5] = scaler.transform(X_train[:, :5])

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    logger.info(f"Tập huấn luyện: {len(X_train)} mẫu, Tập kiểm tra: {len(X_test)} mẫu")

    # Huấn luyện mô hình với Grid Search
    model = XGBRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    logger.info(f"Tham số tốt nhất: {grid_search.best_params_}")
    logger.info(f"R² CV tốt nhất: {grid_search.best_score_:.4f}")

    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test, y_pred)
    mean_price = y_test_original.mean()
    error_ratio = mae / mean_price
    logger.info(f"Dataset sau xử lý có {len(X_train)} mẫu huấn luyện.")
    logger.info(f"Độ lệch tuyệt đối trung bình (MAE) trên tập kiểm tra: {mae:,.0f} VND")
    logger.info(f"R² Score trên tập kiểm tra: {r2:.4f}")
    logger.info(f"Giá trung bình: {mean_price:,.0f} VND")
    logger.info(f"Tỷ lệ sai số: {error_ratio:.2%}")

    # Vẽ biểu đồ dự đoán so với thực tế
    plt.figure(figsize=(8, 4))
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
    plt.xlabel("Giá thực tế (VND)")
    plt.ylabel("Giá dự đoán (VND)")
    plt.title("Dự đoán so với Giá trị thực tế")
    plt.savefig("prediction_vs_actual.png")
    plt.close()
    logger.info("Đã tạo prediction_vs_actual.png")

    return model, X_test, y_test, df, scaler, main_door_encoder, balcony_encoder, legal_doc_encoder

# Định nghĩa API
class HouseInput(BaseModel):
    location: str
    type_of_house: str
    land_area: str
    bedrooms: Optional[str] = None
    toilets: Optional[str] = None
    total_floors: Optional[str] = None
    main_door_direction: Optional[str] = None
    balcony_direction: Optional[str] = None
    legal_documents: Optional[str] = None

class House(BaseModel):
    location: Optional[str] = None
    type_of_house: Optional[str] = None
    land_area: Optional[str] = None
    bedrooms: Optional[str] = None
    toilets: Optional[str] = None
    total_floors: Optional[str] = None
    main_door_direction: Optional[str] = None
    balcony_direction: Optional[str] = None
    legal_documents: Optional[str] = None

class HistoryInput(BaseModel):
    history: list

@app.api_route("/predict", methods=["POST", "OPTIONS"])
async def predict_price(house: HouseInput):
    global model, scaler, main_door_encoder, balcony_encoder, legal_doc_encoder, df
    if model is None or scaler is None or main_door_encoder is None or balcony_encoder is None or legal_doc_encoder is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa được khởi tạo")

    try:
        # Phân tích đầu vào
        area = extract_area(house.land_area)
        if pd.isna(area):
            raise ValueError("Diện tích đất không hợp lệ")

        bedrooms = extract_room(house.bedrooms) if house.bedrooms else 0
        toilets = extract_room(house.toilets) if house.toilets else 0
        floors = extract_floors(house.total_floors) if house.total_floors else 1
        category_vec = encode_category(house.type_of_house)
        district_code = encode_location(house.location)
        main_door_vec = encode_direction(house.main_door_direction or 'unknown', main_door_encoder)
        balcony_vec = encode_direction(house.balcony_direction or 'unknown', balcony_encoder)
        legal_doc_vec = encode_legal_documents(house.legal_documents or 'unknown', legal_doc_encoder)

        # Tính giá trên mỗi m² trung bình từ dataset
        price_per_m2 = df['Price_per_m2'].mean() if 'Price_per_m2' in df and not df['Price_per_m2'].empty else 1e8

        # Tạo vector đặc trưng
        expected_feature_count = 5 + 8 + 1 + 9 + 9 + 7
        features = (
            [area, bedrooms, toilets, floors, price_per_m2] +
            category_vec +
            [district_code] +
            main_door_vec +
            balcony_vec +
            legal_doc_vec
        )
        if len(features) != expected_feature_count:
            raise ValueError(f"Chiều dài đặc trưng không đúng: {len(features)} (dự kiến {expected_feature_count})")

        features = np.array([features])

        # Chuẩn hóa đặc trưng số
        features[:, :5] = scaler.transform(features[:, :5])

        # Dự đoán
        log_price = model.predict(features)[0]
        price = np.expm1(log_price)

        return {"predicted_price_vnd": f"{price:,.0f} VND"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")

@app.post("/save-history")
async def save_history(data: HistoryInput):
    try:
        with open("prediction_history.json", "w", encoding="utf-8") as f:
            json.dump(data.history, f, ensure_ascii=False, indent=2)
        return {"message": "Lịch sử đã được lưu thành công"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu lịch sử: {str(e)}")

@app.get("/get-history")
async def get_history():
    try:
        with open("prediction_history.json", "r", encoding="utf-8") as f:
            history = json.load(f)
        return history
    except FileNotFoundError:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc lịch sử: {str(e)}")

# Khởi tạo mô hình khi server bắt đầu
@app.on_event("startup")
async def startup_event():
    global model, X_test, y_test, df, scaler, main_door_encoder, balcony_encoder, legal_doc_encoder
    model, X_test, y_test, df, scaler, main_door_encoder, balcony_encoder, legal_doc_encoder = initialize_data_and_model()
    logger.info("Mô hình và dữ liệu đã được khởi tạo thành công")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)