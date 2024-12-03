import numpy as np
import pandas as pd
import re
import json
import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import OneHotEncoder
from joblib import load

# Выгрузка модели
ridge_modif = load('ridge_modif.joblib')
print(ridge_modif)

# Функция для предобработки данных
PATH_TRAIN = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'

def extract_numbers(column):
    if pd.notnull(column):
        numbers = re.findall(r'-?\d+\.?\d*', column)
        return float(numbers[0]) if numbers else np.nan
    else: return column

def DataPreprocessing(df_test):
    df_train = pd.read_csv(PATH_TRAIN)

    # Удаление столбца torque
    df_train = df_train.drop(labels='torque', axis=1)
    df_test = df_test.drop(labels='torque', axis=1)

    # Прежде чем уберем единицы измерения столбцов, мы их вынесем в названия колонок
    df_train.rename(columns={'mileage': 'mileage, [kmpl]', 'engine': 'engine, [CC]', "max_power": "max_power, [bhp]"},
                    inplace=True)
    df_test.rename(columns={'mileage': 'mileage, [kmpl]', 'engine': 'engine, [CC]', "max_power": "max_power, [bhp]"},
                   inplace=True)

    # Удаление единиц измерения
    # Тренировочные данные
    df_train['mileage, [kmpl]'] = df_train['mileage, [kmpl]'].apply(extract_numbers)
    df_train['engine, [CC]'] = df_train['engine, [CC]'].apply(extract_numbers)
    df_train['max_power, [bhp]'] = df_train['max_power, [bhp]'].apply(extract_numbers)
    # Тестовые данные
    df_test['mileage, [kmpl]'] = df_test['mileage, [kmpl]'].apply(extract_numbers)
    df_test['engine, [CC]'] = df_test['engine, [CC]'].apply(extract_numbers)
    df_test['max_power, [bhp]'] = df_test['max_power, [bhp]'].apply(extract_numbers)

    # создаем копии исходных таблиц
    fill_df_train = df_train.copy()
    fill_df_test = df_test.copy()
    # Создаем словарь из столбца: число(признак) на который надо заменить пропуски
    values = {
        'mileage, [kmpl]': fill_df_train['mileage, [kmpl]'].median(),
        'engine, [CC]': fill_df_train['engine, [CC]'].median(),
        'max_power, [bhp]': fill_df_train['max_power, [bhp]'].median(),
        'seats': fill_df_train['seats'].median()
    }
    # Заполяняем пропуски в соответствии с заявленным словарем
    fill_df_train = fill_df_train.fillna(values)
    fill_df_test = fill_df_test.fillna(values)

    # Удаление дубликатов
    X = fill_df_train.drop('selling_price', axis=1)
    train_dupl_columns = list(X.columns)
    mask_train = X.duplicated(subset=train_dupl_columns)
    df_train_duplicates = X[mask_train]
    df_train_unique = fill_df_train.drop_duplicates(subset=train_dupl_columns)
    df_train_unique = df_train_unique.reset_index(drop=True)

    #X = fill_df_test.drop('selling_price', axis=1)
    #test_dupl_columns = list(X.columns)
    #mask_test = X.duplicated(subset=test_dupl_columns)
    #df_test_duplicates = X[mask_test]
    # Удаляем дубликаты
    #df_test_unique = fill_df_test.drop_duplicates(subset=test_dupl_columns)
    # Обновление индексов строк
    #df_test_unique = df_test_unique.reset_index(drop=True)

    df_test_unique = fill_df_test.copy() # Не стал удалять дубликаты на тесте

    # Предопределение типов на обучающей выборке
    df_train_unique['engine, [CC]'] = df_train_unique['engine, [CC]'].astype(int)
    df_train_unique['seats'] = df_train_unique['seats'].astype(int)

    # Предопределение типов на тестовой выборке
    df_test_unique['engine, [CC]'] = df_test_unique['engine, [CC]'].astype(int)
    df_test_unique['seats'] = df_test_unique['seats'].astype(int)

    df_train = df_train_unique.copy()
    df_test = df_test_unique.copy()

    # Удаление столбца name
    df_train = df_train.drop(columns=["name"])
    df_test = df_test.drop(columns=["name"])

    # Зададим обучающую выборку, удалив в ней целевую переменную
    X_train_cat = df_train.copy()
    X_train_cat = X_train_cat.drop(columns=["selling_price"])
    X_test_cat = df_test.copy()
    X_test_cat = X_test_cat.drop(columns=["selling_price"])

    cat_features = ["fuel", "seller_type", "transmission", "owner", "seats"]
    encoded_train_features = []
    encoded_test_features = []
    for feature in cat_features:
        encoded = OneHotEncoder()
        train_enc = encoded.fit_transform(X_train_cat[feature].values.reshape(-1, 1)).toarray()
        test_enc = encoded.transform(X_test_cat[feature].values.reshape(-1, 1)).toarray()
        n = X_train_cat[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_train_df = pd.DataFrame(train_enc, columns=cols)
        encoded_test_df = pd.DataFrame(test_enc, columns=cols)
        encoded_train_df.index = X_train_cat.index
        encoded_test_df.index = X_test_cat.index
        encoded_train_features.append(encoded_train_df)
        encoded_test_features.append(encoded_test_df)

    X_train_cat = pd.concat([X_train_cat, *encoded_train_features[:]], axis=1)
    X_test_cat = pd.concat([X_test_cat, *encoded_test_features[:]], axis=1)

    # Удаление ненужных столбцов на трейне
    drop_cols = ["fuel", "seller_type", "transmission", "owner", "seats", "fuel_1", "seller_type_1",
                 "transmission_1", "owner_1", "seats_1"]
    X_train_cat.drop(columns=drop_cols, inplace=True)
    X_test_cat.drop(columns=drop_cols, inplace=True)

    return X_test_cat



class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

app = FastAPI()


# На вход подаются признаки 1-го объекта, на выходе - прогноз
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([json.loads(item.model_dump_json())])
    df_1 = DataPreprocessing(data)
    pred_test_ridge_modif = ridge_modif.predict(df_1)
    return pred_test_ridge_modif[0]

# На входе - файл .csv с объектами, на выходе - предсказание для каждого объекта
@app.post("/predict_items", response_class=StreamingResponse)
async def predict_items_csv(file: UploadFile = File(...)):
    content = await file.read()
    data = pd.read_csv(io.BytesIO(content))
    df_1 = DataPreprocessing(data)
    pred_test_ridge_modif = ridge_modif.predict(df_1)
    #df_1['predictions_price'] = pred_test_ridge_modif

    data['predictions_price'] = pred_test_ridge_modif

    output_stream = io.StringIO()
    #df_1.to_csv(output_stream, index=False)

    data.to_csv(output_stream, index=False)

    output_stream.seek(0)
    response = StreamingResponse(iter([output_stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions_output.csv"
    return response
