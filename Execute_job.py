import logging
import threading
import time
import mysql.connector as mariadb

def update(id):
    mariadb_connection = mariadb.connect(host="localhost", user='root', password='1234', database='jobs')
    cursor = mariadb_connection.cursor()
    sql = "UPDATE tb_jobs SET status = %s WHERE id = %s"
    val = ('2', id)
    cursor.execute(sql, val)
    mariadb_connection.commit()
    print('update ID ', id)


def thread_function(id):
    logging.info("Thread : starting.....!!")
    time.sleep(120)
    update(id)
    logging.info("Thread : finishing.....!!")

def execute_job(id):
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    threads = list()
    logging.info("Main    : create and start thread.")
    x = threading.Thread(target=thread_function, args=(id,))
    threads.append(x)
    x.start()
