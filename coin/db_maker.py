import sqlite3

import os
import pandas as pd

from coin import C
from datetime import datetime


class DbMaker:
    __conn = None

    @staticmethod
    def __get_conn():
        if DbMaker.__conn is None:
            DbMaker.__conn = sqlite3.connect(C['db_path'])
        return DbMaker.__conn

    @staticmethod
    def create_table(table_name):
        conn = DbMaker.__get_conn()
        conn.execute('DROP TABLE IF EXISTS {table_name}'.format(table_name=table_name))
        sql = """
            CREATE TABLE IF NOT EXISTS {table_name} (
                time DATETIME NOT NULL UNIQUE,
                close REAL,
                close_forecast REAL,
                modified_time DATETIME DEFAULT (datetime('now','localtime'))
            )
        """.format(table_name=table_name)
        conn.execute(sql)
        # trigger：更新row时自动更新modified_time
        sql = """
            CREATE TRIGGER trigger_{table_name}
                UPDATE OF 
                    close, 
                    close_forecaset
                ON {table_name}
            BEGIN
                UPDATE {table_name} SET modified_time=(datetime('now','localtime')) WHERE time=OLD.time;
            END
        """.format(table_name=table_name)
        conn.execute(sql)
        conn.commit()

    @staticmethod
    def update_table(table_name, close_df, close_forecast_df):
        conn = DbMaker.__get_conn()

        if close_df is not None:
            close_df.to_sql('temp', conn, if_exists='replace')
            # 插入空缺的close历史数据，已存在则更新（下一个sql）
            sql = """
                INSERT OR IGNORE INTO {table_name} ('time', 'close', 'close_forecast')
                SELECT
                    time,
                    close,
                    close_forecast
                FROM temp
            """.format(table_name=table_name)
            conn.execute(sql)
            # 更新已有的close历史数据，此时对应的close_forecast数据已不变
            sql = """
                UPDATE {table_name}
                SET close = (
                    SELECT close 
                    FROM temp
                    WHERE temp.time = {table_name}.time 
                    AND close IS NOT NULL 
                )
            """.format(table_name=table_name)
            conn.execute(sql)

        if close_forecast_df is not None:
            close_forecast_df.to_sql('temp', conn, if_exists='replace')
            # 插入close_forecast数据，已存在则更新，此时close数据为空
            sql = """
                INSERT OR REPLACE INTO {table_name} ('time', 'close', 'close_forecast')
                SELECT
                  time,
                  close,
                  close_forecast
                FROM temp
            """.format(table_name=table_name)
            conn.execute(sql)

        conn.execute('DROP TABLE temp')
        conn.commit()
        # df.to_sql(name=table_name, con=conn, if_exists='append', chunksize=1000, index=False)

    @staticmethod
    def dump_table(table_name, limit=-1):
        conn = DbMaker.__get_conn()
        cursor = conn.cursor()

        sql = """
            SELECT 
                time, close, close_forecast
            FROM (
                SELECT 
                    time, close, close_forecast
                FROM
                    {table_name}
                ORDER BY time DESC 
                {limit_clause}
            )
            ORDER BY time ASC 
        """.format(
            table_name=table_name,
            limit_clause='' if limit == -1 else 'LIMIT {}'.format(limit)
        )

        df = pd.read_sql_query(sql, conn)
        df['close'] = df['close'].apply(lambda x: '-' if x is None or x == '' else x)
        df['close_forecast'] = df['close_forecast'].apply(lambda x: '-' if x is None or x == '' else x)
        df['time'] = df['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M'))
        df.columns = ['date', 'close', 'close_forecast']
        df.to_csv(path_or_buf=os.path.join(C['ui_data'], table_name + '.csv'), index=False)
        print(df)


if __name__ == '__main__':
    # DbMaker.create_table('usdt_btc_180101_p5_i256_o16_f2')
    DbMaker.dump_table('usdt_btc_150101_p5_i256_o16_f2')
