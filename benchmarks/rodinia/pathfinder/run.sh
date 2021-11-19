rm orig*.txt
rm mod*.txt

for i in {1..10}
do
  ./pathfinder_orig.bin 1000000 200 20 2>>orig200.txt
  ./pathfinder_orig.bin 1000000 600 20 2>>orig600.txt
  ./pathfinder_orig.bin 1000000 1000 20 2>>orig1000.txt

  ./pathfinder_mod.bin 1000000 200 20 2>>mod200.txt
  ./pathfinder_mod.bin 1000000 600 20 2>>mod600.txt
  ./pathfinder_mod.bin 1000000 1000 20 2>>mod1000.txt
done	

