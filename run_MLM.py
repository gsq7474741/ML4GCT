from __future__ import annotations

import datetime
import sqlite3
from typing import List, Dict

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from prettytable import PrettyTable
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sqlmodel import Field, Session, SQLModel, create_engine, select
from tqdm import tqdm


class Record(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    seed: int
    target: str
    model: str
    rmse: float
    r2: float
    custom_str: str | None


class TTestResult(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    target: str
    model1: str
    model2: str
    t_value: float
    p_value: float
    custom_str: str | None


def train_model(train_set, test_set, target: str, reg_metrics: List[str], seed: int, root: str):
    path = f'{root}/regression-{target}'

    predictor = TabularPredictor(label=target, path=path,
                                 verbosity=0,
                                 problem_type='regression'
                                 ).fit(train_set,
                                       hyperparameters={
                                           "LR": {},
                                           "GBM": {},
                                           "CAT": {},
                                           "SIMPLE_ENS_WEIGHTED": [
                                               {'use_orig_features': False,
                                                'max_base_models': 25,
                                                'max_base_models_per_type': 5,
                                                "save_bag_folds": True}
                                           ]
                                       },
                                       )
    leader_board = predictor.leaderboard(test_set, silent=True, extra_metrics=reg_metrics)

    for model_record in range(leader_board.shape[0]):
        yield Record(seed=seed,
                     target=target,
                     model=leader_board.iloc[model_record]["model"],
                     rmse=-leader_board['root_mean_squared_error'].values[model_record],
                     r2=leader_board['r2'].values[model_record]
                     )


def preprocess_data(file_path: str, features: list, targets: list) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='utf-8')
    df.fillna(0, inplace=True)
    df[features] = power_transform(df[features])
    df[targets] = MinMaxScaler().fit_transform(df[targets])

    return df


def train_and_store_models(df: pd.DataFrame, features: list, targets: list, reg_metrics: List[str], seeds: np.ndarray,
                           engine,
                           test_size: float):
    data_induce = np.arange(0, len(df))
    pbar = tqdm(total=len(seeds) * len(targets))
    for seed in seeds:
        for target in targets:
            np.random.seed(seed)
            train_index, val_index = train_test_split(data_induce, test_size=test_size, random_state=seed)
            train_set = df.iloc[train_index][features + [target]]
            test_set = df.iloc[val_index][features + [target]]
            records = []
            for record in train_model(train_set, test_set, target, reg_metrics, seed=seed, root=f'ag-models'):
                records.append(record)
            with Session(engine) as session:
                session.add_all(records)
                session.commit()
            pbar.update(1)
    pbar.close()


def perform_t_tests(engine, targets: list, models: list):
    with Session(engine) as session:
        for target in targets:
            for model_1 in models:
                for model_2 in models:
                    res1 = query_metrics(session, model_1, target, 'r2')
                    res2 = query_metrics(session, model_2, target, 'r2')
                    t, p = ttest_ind(res1, res2, equal_var=False)
                    session.add(TTestResult(target=target, model1=model_1, model2=model_2, t_value=t, p_value=p,
                                            custom_str=None))
                    session.commit()


def query_metrics(session: Session, model: str, target: str, metric: str) -> np.ndarray:
    rev_short_names = {
        'root_mean_squared_error': 'rmse',
        'r2': 'r2'
    }
    metric = rev_short_names[metric]
    statement = select(Record).where(Record.model == model, Record.target == target)
    results = session.exec(statement).all()
    res = np.zeros(len(results))
    for idx, record in enumerate(results):
        res[idx] = getattr(record, metric)
    return res


def print_mean_metrics(engine, targets: List[str], models: List[str], reg_metrics: List[str],
                       short_names: Dict[str, str]):
    with Session(engine) as session:
        title = ['']
        for model in models:
            for metric in reg_metrics:
                title.append(f'{short_names[model]}_{short_names[metric]}')
        td = PrettyTable(title)
        for target in targets:
            row = [target]
            for model in models:
                for metric in reg_metrics:
                    res = query_metrics(session, model, target, metric)
                    row.append(f'{np.mean(res):.4f}({np.std(res):.4f})')
            td.add_row(row)
        print(td)


def print_t_tests(engine, targets: List[str], models: List[str], short_names: Dict[str, str]):
    with Session(engine) as session:
        for target in targets:
            title = [target] + [short_names[model] for model in models]
            td = PrettyTable(title)
            for model1 in models:
                row = [short_names[model1]]
                for model2 in models:
                    res = session.exec(
                        select(TTestResult).where(
                            TTestResult.target == target,
                            TTestResult.model1 == model1,
                            TTestResult.model2 == model2,
                        )).one()
                    sign = ''
                    if res.p_value < 0.001:
                        sign = '***'
                    elif res.p_value < 0.01:
                        sign = '**'
                    elif res.p_value < 0.05:
                        sign = '*'
                    row.append(f'{res.t_value:.4f}{sign}')
                td.add_row(row)
            print(td)


def main():
    # 设置实验参数
    np.random.seed(42)
    run_times = 100
    test_size = 0.3
    dump_db = True
    seeds = np.random.randint(0, 100000, run_times)

    features = [
        'A1', 'A2', 'A3', 'A4',
        'B1', 'B2', 'B3', 'B4',
        'C1', 'C2',
    ]
    targets = [
        'Total Score',
        'Fluency',
        'Originality',
        'Flexibility'
    ]
    models = [
        'CatBoost',
        'WeightedEnsemble_L2',
        'LinearModel',
        'LightGBM',
    ]
    reg_metrics = ['root_mean_squared_error',
                   'r2']
    short_names = {
        'CatBoost': 'CB',
        'WeightedEnsemble_L2': 'WE',
        'LinearModel': 'LR',
        'LightGBM': 'LGBM',
        'root_mean_squared_error': 'rmse',
        'r2': 'r2'
    }

    # 数据预处理
    df = preprocess_data('data/Process data and Scores.csv', features, targets)

    # 连接数据库
    sqlite_file_name = "experiment.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"
    engine = create_engine(sqlite_url)
    SQLModel.metadata.create_all(engine)
    session = Session(engine)

    # 判断是否存在记录
    if session.exec(select(Record)).first() is not None:
        # 询问使用原有记录还是重新训练
        choice = input("Use existing records? (y/n)")
        assert choice in ['y', 'n'], "Invalid choice"
        if choice == 'y':
            pass
        else:
            # 删除原有table
            Record.__table__.drop(engine)
            TTestResult.__table__.drop(engine)
            session.commit()
            SQLModel.metadata.create_all(engine)
            # 训练模型并存储结果
            train_and_store_models(df, features, targets, reg_metrics, seeds, engine, test_size)
            # 进行t检验
            perform_t_tests(engine, targets, models)
    else:
        # 训练模型并存储结果
        train_and_store_models(df, features, targets, reg_metrics, seeds, engine, test_size)
        # 进行t检验
        perform_t_tests(engine, targets, models)

    # 打印平均指标
    print_mean_metrics(engine, targets, models, reg_metrics, short_names)

    # 打印t检验结果
    print_t_tests(engine, targets, models, short_names)

    # 保存数据库
    gen_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if dump_db:
        # 连接到SQLite数据库
        conn = sqlite3.connect(sqlite_file_name)
        # 执行SQL查询并将结果存储到DataFrame中
        df_export = pd.read_sql_query("SELECT * FROM record", conn)
        df_export.to_csv(f'dump/records_{gen_time}.csv', index=False)
        df_export = pd.read_sql_query("SELECT * FROM ttestresult", conn)
        df_export.to_csv(f'dump/t_test_results_{gen_time}.csv', index=False)


if __name__ == '__main__':
    main()
