from syncHDF import create_syncHDF
import os

TDT_tank = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/Mario - Neural Data/'
os.chdir(TDT_tank)
create_syncHDF('Mario20181029',TDT_tank)

for directories in os.listdir(os.getcwd())[11:-3]:
	filename = directories
	print(filename)
	create_syncHDF(filename, TDT_tank)


# syncHDF didn't work for Luigi 20181029, 20181120; Mario 20181010;
# stopped running at Mario20181018-1
#Luigi best days: 20181030, 20181118, 20181117
# Mario best days: 20181017, 20181027, 20181013, 20181101