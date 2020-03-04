
import os
from sys import platform
try:
    # Windows
    if platform == "win32":
        script_dir = os.path.dirname(__file__)
        PREDICRFILE = script_dir + "//Storage//Model//"
        READFILE_EXCEL = script_dir + "//Storage//Excel//"
        READFILE_EXCEL_DATA = script_dir + "//Storage//Excel//data//"
        READFILE = script_dir + "//Storage//Excel//bigfarm-ml-all.xlsx"
        READFILE_BREED = script_dir + "//Storage//Excel//breed.xlsx"
        READFILE_DATA = script_dir + "//Storage//Excel//alldata_dataframe.csv"
        READFILE_PARAMETER = script_dir + '//Storage//Parameter//'
        READFILE_CLUSTER = script_dir + '//Storage/Cluster/'
        READFILE_IMAGE = script_dir +"//Storage//Image//"
    else:

        READFILE_IMAGE = "Storage/Image/"
        READFILE_DATA = "Storage/Excel/alldata_dataframe.csv"
        READFILE_EXCEL = "Storage/Excel/"
        READFILE_EXCEL_DATA = "Storage/Excel/data/"
        PREDICRFILE = "Storage/Model/"
        READFILE = "Storage/Excel/bigfarm-ml-all.xlsx"
        READFILE_BREED = "Storage/Excel/breed.xlsx"
        READFILE_DATA = "Storage/Excel/alldata_dataframe.csv"
        READFILE_PARAMETER = 'Storage/Parameter/'
        READFILE_CLUSTER = 'Storage/Cluster/'

except ImportError as e:
    print('Error:')
    raise e