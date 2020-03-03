import mysql.connector as mariadb
import execute_job2 as Execute_job

mariadb_connection = mariadb.connect(host="localhost", user='root', password='1234', database='jobs')
cursor = mariadb_connection.cursor()

cursor.execute("SELECT * FROM tb_jobs where status IN (0,1)  ORDER BY time_reg ASC LIMIT 1")
for data in cursor:
    if data[4] == '0':
        Execute_job.execute_job(data[0])
    if data[4] == '1':
        print('Job is processing stop now')



