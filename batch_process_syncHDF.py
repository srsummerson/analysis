from syncHDF import create_syncHDF
import os

TDT_tank = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/Mario - Neural Data/'
os.chdir(TDT_tank)

files = os.listdir(os.getcwd())[11:-3]
#files = ['Luigi20181120']
files = ['Mario20181025', 'Mario20181025-1', 'Mario20181027','Mario20181027-1', 'Mario20181101', 'Mario20181101-1', 'Mario20181105','Mario20181105-1']

for directories in files:
	filename = directories
	print(filename)
	create_syncHDF(filename, TDT_tank)

# syncHDF didn't work for Luigi 20181029, 20181120; Mario 20181010;
# stopped running at Mario20181018-1
#Luigi best days: 20181030, 20181118, 20181117
# Mario best days: 20181017, 20181027, 20181013, 20181101