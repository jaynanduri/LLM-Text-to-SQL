import os
from pyrfc import Connection
os.environ['SAPNWRFC_HOME'] = '/usr/local/sap/nwrfcsdk'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/sap/nwrfcsdk/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['DYLD_LIBRARY_PATH'] = '/usr/local/sap/nwrfcsdk/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')
# Define connection parameters
conn_params = {
    'ashost': '172.16.1.64',      # SAP Application Server host
    'sysnr': '00',             # SAP System Number
    'client': '100',           # SAP Client
    'user': 'SAPUSER',        # SAP Username
    'passwd': 'India@123',      # SAP Password
    'lang': 'EN',              # Language
    'trace': '3',              # Trace level for RFC debugging (optional)
}

# Establish connection
conn = Connection(**conn_params)

# Check if connection is established
if conn.alive:
    print("Connection to SAP HANA successful")
else:
    print("Failed to connect to SAP HANA")